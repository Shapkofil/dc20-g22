import torch

from dc2_g22.lstm.model import LSTMForcaster
from dc2_g22.lstm.train_test import train_test_loop

def find_device():
    DEBUG = False
    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
    return device
 

def main():
    n_wards = 24
    
    device = find_device()
    model = LSTMForcaster(5, 24, 24, 3, device)
    model.to(device)
    raise NotImplemented
    # TODOOOOO implement the Dataset and loaders.
    train_test_loop(model,
                    testloader,
                    trainloader,
    )

def start():
    """
    Starts the little framework we have here.
    """
    main()



if __name__ == "__main__":
    start()
