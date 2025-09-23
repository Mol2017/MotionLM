# MotionLM: Multi-Agent Motion Forecasting as Language Modeling

MotionLM represents continuous trajectories as sequences of discrete motion tokens and casts multi-agent motion prediction as a language modeling task. This repository contains a reproduction of the MotionLM model from the paper:

**"MotionLM: Multi-Agent Motion Forecasting as Language Modeling"**  
*Ari Seff, Brian Cera, Dian Chen, Mason Ng, Aurick Zhou, Nigamaa Nayakanti, Khaled S. Refaat, Rami Al-Rfou, Benjamin Sapp*  
International Conference on Computer Vision (ICCV) 2023  
[Paper Link](https://arxiv.org/abs/2309.16534)

## Reproduced Components

This reproduction implements the core components of MotionLM:

### 🎯 Motion Tokenizer (`models/motion_tokenizer.py`)
- Verlet-wrapped motion tokenizer that encodes trajectory corrections as discrete tokens
- Uses a BxB grid to quantize small corrections (dx_corr, dy_corr)
- Implements Verlet integration: Δ_t = Δ_{t-1} + δ_t

### 🏛️ Scene Encoder (`models/scene_encoder.py`)
- Processes multi-modal scene context including:
  - Agent historical trajectories
  - Lane information 
  - Traffic light states
- Outputs scene memory tokens for conditioning the decoder

### 🔮 Motion Decoder (`models/motion_decoder.py`)
- Transformer-based autoregressive decoder
- Generates motion tokens conditioned on scene memory
- Supports both training (teacher forcing) and inference modes

### 🧠 MotionLM Model (`models/motionlm.py`)
- Complete end-to-end model combining scene encoder and motion decoder
- Implements training and inference workflows
- Supports multi-agent joint prediction

## Quick Start

### Requirements
- Python 3.7+
- PyTorch
- Tensorflow

### Running the Model

To test the reproduced model components:

```bash
python test/model_test.py

python test/tokenizer_test.py
```

This test script will:
1. Test the scene encoder with sample agent histories, lanes, and traffic lights
2. Test the motion decoder with scene memory and token sequences  
3. Test the complete MotionLM model with end-to-end training and inference
4. Print model parameter counts and sample outputs

## Model Architecture

```
Input Scene Data → Scene Encoder → Scene Memory
                                        ↓
                   Motion Tokens → Motion Decoder → Next Token Logits
```

### Key Features:
- **Discrete Motion Tokens**: Continuous trajectories tokenized using Verlet integration
- **Multi-Modal Scene Context**: Joint encoding of agents, lanes, and traffic signals
- **Autoregressive Generation**: Sequential token prediction for future trajectories
- **Multi-Agent Modeling**: Joint prediction of interacting agent futures

## Configuration

Model hyperparameters can be adjusted in `models/config.py`:

- `d_model`: Token embedding dimension (default: 256)
- `vocab_size`: Motion token vocabulary size (default: 169)
- `bins_per_coord`: Discretization bins per coordinate (default: 13)
- `max_corr`: Maximum correction magnitude (default: 1.5)

## Acknowledgments

This reproduction is based on the original MotionLM paper by the Waymo team. 
```bibtex
@article{seff2023motionlm,
  title={MotionLM: Multi-Agent Motion Forecasting as Language Modeling},
  author={Seff, Ari and Cera, Brian and Chen, Dian and Ng, Mason and Zhou, Aurick and Nayakanti, Nigamaa and Refaat, Khaled S. and Al-Rfou, Rami and Sapp, Benjamin},
  journal={arXiv preprint arXiv:2309.16534},
  year={2023}
}
```
