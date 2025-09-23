import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all logs, 1=filter INFO, 2=filter INFO+WARNING, 3=only ERRORs


import torch
import argparse
import tensorflow as tf


def test_torch_and_gpu():
    # Check PyTorch installation
    print("PyTorch version:", torch.__version__)

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Running on CPU.")

    # Test tensor creation and GPU computation
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        print("Tensor created on:", device)
        print("Tensor:", tensor)
    except Exception as e:
        print("Error during tensor creation or GPU computation:", e)


def test_tensorflow_and_gpu():
    # Check TensorFlow installation
    print("TensorFlow version:", tf.__version__)

    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Number of GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
    else:
        print("No GPU available. Running on CPU.")

    # Test tensor creation and GPU computation
    try:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            tensor = tf.constant([1.0, 2.0, 3.0])
            print("Tensor created on:", "GPU" if gpus else "CPU")
            print("Tensor:", tensor)
    except Exception as e:
        print("Error during tensor creation or GPU computation:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_pytorch", action="store_true", help="Test PyTorch and GPU")
    ap.add_argument("--test_tensorflow", action="store_true", help="Test TensorFlow and GPU")
    args = ap.parse_args()
    
    # If no flags are specified, test both
    if not args.test_pytorch and not args.test_tensorflow:
        args.test_pytorch = True
        args.test_tensorflow = True
    
    if args.test_pytorch:
        print("=== Testing PyTorch ===")
        test_torch_and_gpu()
        print()
    
    if args.test_tensorflow:
        print("=== Testing TensorFlow ===")
        test_tensorflow_and_gpu()

    print(torch.__version__)          # e.g. '2.3.1+cu121' or '2.2.0+cpu'
    print(torch.version.cuda)         # e.g. '12.1' (build CUDA)
    print(torch.backends.cudnn.version())  