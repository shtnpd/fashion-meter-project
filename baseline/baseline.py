import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
import wandb

# === Параметры ===
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Инициализация W&B
wandb.init(project="cifar10-baseline", config={
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "model": "ResNet18",
    "dataset": "CIFAR-10"
})

# === Трансформации ===

# Аугментации данных
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Рандомное обрезание
    transforms.RandomHorizontalFlip(),     # Горизонтальное отражение
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Нормализация
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# === Загрузка CIFAR-10 ===
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# === Модель ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 классов для CIFAR-10
model = model.to(DEVICE)

# === Оптимизатор и функция потерь ===
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === Функции для обучения и оценки ===
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs} started...")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Логирование в W&B
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Логирование в W&B
    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
    return accuracy

# === Основной процесс ===
if __name__ == "__main__":
    print("=== Начало обучения на CIFAR-10 ===")
    train_model(model, train_loader, optimizer, criterion, EPOCHS)

    print("=== Оценка модели ===")
    evaluate_model(model, test_loader)

    print("=== Сохранение модели ===")
    torch.save(model.state_dict(), "baseline_model_cifar10.pth")
    print("Baseline модель успешно сохранена!")