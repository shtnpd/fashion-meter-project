import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import wandb

from dataset import SafeDataset, collate_fn
from model import create_custom_model
from train import train_model
from evaluate import evaluate_model

# === Параметры ===
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Логирование W&B ===
wandb.init(project="custom-dataset-training-and-benchmark", name="custom_model_run")

# === Трансформации ===
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === Датасет ===
TRAIN_CSV = "./classes/train.csv"
TEST_CSV = "./classes/test.csv"
ROOT_DIR = "./classes"

train_dataset = SafeDataset(TRAIN_CSV, ROOT_DIR, transform=transform_train)
test_dataset = SafeDataset(TEST_CSV, ROOT_DIR, transform=transform_test)

# === Загрузка данных ===
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

# === Модель ===
custom_model = create_custom_model(len(train_dataset.classes), DEVICE)

# === Оптимизатор и функция потерь ===
optimizer = optim.Adam(custom_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === Обучение ===
print("=== Обучение кастомной модели ===")
train_model(custom_model, train_loader, optimizer, criterion, DEVICE, EPOCHS)

# === Оценка ===
print("=== Оценка кастомной модели ===")
custom_accuracy, custom_report = evaluate_model(custom_model, test_loader, train_dataset.classes, DEVICE)
print(f"Кастомная модель, Точность: {custom_accuracy:.2f}%")
print(custom_report)
wandb.log({"model": "Custom ResNet18", "accuracy": custom_accuracy, "classification_report": custom_report})

# === Сохранение модели ===
print("=== Сохранение кастомной модели ===")
torch.save(custom_model.state_dict(), "custom_model.pth")