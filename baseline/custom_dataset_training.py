# болванка
import os
import torch
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# === Параметры ===
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
ROOT_DIR = "../classes/"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test.csv")

# === Трансформации ===
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === Dataset ===
class FashionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(self.data['folder'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        folder, filename = f"{row['folder']}-clean-rescaled", row['filename']
        img_path = os.path.join(self.root_dir, folder, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Файл {img_path} не найден!")
        image = transforms.functional.pil_to_tensor(
            transforms.functional.to_pil_image(img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[folder]
        return image, label

# === DataLoader ===
train_dataset = FashionDataset(TRAIN_CSV, ROOT_DIR, transform=transform_train)
test_dataset = FashionDataset(TEST_CSV, ROOT_DIR, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# === Модель ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(DEVICE)

# === Оптимизатор и функция потерь ===
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === Функции для обучения и оценки ===
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
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

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# === Основной процесс ===
if __name__ == "__main__":
    print("=== Начало обучения на кастомном датасете ===")
    train_model(model, train_loader, optimizer, criterion, EPOCHS)

    print("=== Оценка модели ===")
    evaluate_model(model, test_loader)

    print("=== Сохранение модели ===")
    torch.save(model.state_dict(), "custom_dataset_model.pth")
    print("Модель для кастомного датасета успешно сохранена!")