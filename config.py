import torch

def my_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# === Параметры ===
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = my_device()

# === Пути ===
TRAIN_CSV = "./classes/train.csv"
TEST_CSV = "./classes/test.csv"
ROOT_DIR = "./classes"