import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import MotionLMConfig
from models.scene_encoder import SceneEncoder
from models.motion_decoder import MotionDecoder


class MotionLM(nn.Module):
    """
    MotionLM: A transformer-based model for motion prediction.
    
    The model consists of:
    1. Scene Encoder: Processes agent history, lanes, and traffic lights into scene memory
    2. Motion Decoder: Generates motion tokens autoregressively conditioned on scene memory
    """
    def __init__(self, cfg: MotionLMConfig, F_agent: int, F_lane: int, F_tl: int):
        super().__init__()
        self.cfg = cfg
        self.BOS_ID = cfg.vocab_size  # assuming vocab is [0,V-1], BOS=V
        self.scene_encoder = SceneEncoder(cfg, F_agent, F_lane, F_tl)
        self.motion_decoder = MotionDecoder(cfg)

    def forward_train(self, batch):
        """
        Training forward pass with teacher forcing.
        
        Args:
            batch: Dictionary containing:
                - agents_hist: [B, Na, H, F_agent]
                - lanes: [B, Nl, S, F_lane] 
                - tls: [B, Nl, H, F_tl]
                - agent_hist_valid: [B, Na, H]
                - lane_valid: [B, Nl, S]
                - tl_valid: [B, Nl, H]
                - tokens_gt: [B, T]
                
        Returns:
            loss: Cross-entropy loss
            logits: [B, T, V] predicted token logits
        """
        # Encode scene context
        # [B, Hs, D]
        scene_mem = self.scene_encoder(
            batch['agents_hist'],
            batch['lanes'],
            batch['tls'],
            agent_hist_valid=batch['agent_hist_valid'],
            lane_valid=batch['lane_valid'],
            tl_valid=batch['tl_valid']
        )

        # Prepare teacher forcing inputs
        # tokens_gt: [B, T] - ground truth motion tokens
        tokens_gt = batch['tokens_gt']
        B, T = tokens_gt.shape
        device = tokens_gt.device
        
        # Create BOS token: [B, 1]
        token_bos = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)
        
        # Teacher forcing input: [BOS, token_1, ..., token_{T-1}]
        # [B, T]
        tokens_in = torch.cat([token_bos, tokens_gt[:, :-1]], dim=-1)
        
        # Decode motion tokens
        # [B, T, V]
        logits = self.motion_decoder(tokens_in, scene_mem)
        
        # Compute cross-entropy loss
        # Flatten for loss computation
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B*T, V]
            tokens_gt.reshape(-1),                # [B*T]
        )
        
        return loss, logits

    @torch.no_grad()
    def infer_autoregressive(self, batch, max_length: int = None, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressive inference for motion prediction.
        
        Args:
            batch: Dictionary containing scene context (same as training, except tokens_gt optional)
            max_length: Maximum sequence length to generate (defaults to cfg.T_pred)
            temperature: Sampling temperature (1.0 = no scaling)
            top_k: Top-k sampling (None = no top-k)
            
        Returns:
            tokens: [B, T] generated motion tokens
        """
        # Encode scene context
        # [B, Hs, D]
        scene_mem = self.scene_encoder(
            batch['agents_hist'],
            batch['lanes'],
            batch['tls'],
            agent_hist_valid=batch['agent_hist_valid'],
            lane_valid=batch['lane_valid'],
            tl_valid=batch['tl_valid']
        )
        
        B = scene_mem.shape[0]
        T_pred = max_length if max_length is not None else self.cfg.T
        device = scene_mem.device
        
        # Initialize with BOS token: [B, 1]
        tokens = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)
        
        # Autoregressive generation
        for t in range(T_pred):
            # Get logits for current sequence: [B, t+1, V]
            logits = self.motion_decoder(tokens, scene_mem)
            
            # Get logits for next token: [B, V]
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                top_k = min(top_k, next_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                # Set non-top-k logits to -inf
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Sample next token: [B]
            if temperature == 0.0:
                # Greedy sampling
                next_token = next_logits.argmax(dim=-1)
            else:
                # Multinomial sampling
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Append to sequence: [B, t+2]
            tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
        
        # Remove BOS token and return predictions: [B, T_pred]
        return tokens[:, 1:]

    def forward(self, batch, mode='train', **kwargs):
        """
        Unified forward pass for training and inference.
        
        Args:
            batch: Input batch
            mode: 'train' or 'infer'
            **kwargs: Additional arguments for inference
            
        Returns:
            For training: (loss, logits)
            For inference: tokens
        """
        if mode == 'train':
            return self.forward_train(batch)
        elif mode == 'infer':
            return self.infer_autoregressive(batch, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_metrics(self, batch, predictions=None):
        """
        Compute evaluation metrics.
        
        Args:
            batch: Ground truth batch
            predictions: [B, T] predicted tokens (if None, will generate)
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        if predictions is None:
            predictions = self.infer_autoregressive(batch)
        
        if 'tokens_gt' not in batch or batch['tokens_gt'] is None:
            return {}
        
        tokens_gt = batch['tokens_gt']  # [B, T]
        
        # Token-level accuracy
        correct = (predictions == tokens_gt).float()
        token_accuracy = correct.mean().item()
        
        # Sequence-level accuracy (all tokens correct)
        seq_accuracy = correct.all(dim=-1).float().mean().item()
        
        metrics = {
            'token_accuracy': token_accuracy,
            'sequence_accuracy': seq_accuracy,
        }
        
        return metrics

    def get_parameter_count(self):
        """Get the number of parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }