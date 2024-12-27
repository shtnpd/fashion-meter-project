import time
from pathlib import Path
import torch
from loguru import logger
from tqdm import tqdm
import typer

from fashion_meter_module.config import MODELS_DIR, PROCESSED_DATA_DIR, DEVICE, TEST_CSV
from fashion_meter_module.evaluate import evaluate_model
from fashion_meter_module.model import create_custom_model
from fashion_meter_module.dataset import collate_fn, SafeDataset
from fashion_meter_module.config import TRAIN_CSV, ROOT_DIR, BATCH_SIZE
from torch.utils.data import DataLoader
from torchvision import transforms

app = typer.Typer()

import wandb
from torchvision.transforms import ColorJitter, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize

def train_model(model, train_loader, optimizer, criterion, device, epochs, batch_size, learning_rate):
    """
    Функция для обучения модели.
    """
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    wandb.init(
        project="custom-dataset-training-and-benchmark",
        name=f"custom_model_run-{timestamp}",
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "device": device.type,
        },
    )

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            if inputs.shape[0] == 0:  # Пропускаем пустые данные
                logger.warning("Empty batch encountered, skipping...")
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs} completed. Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    # wandb_logs.finish()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "custom_model.pth",
    train_csv: Path = TRAIN_CSV,
    test_csv: Path = TEST_CSV,
    root_dir: Path = ROOT_DIR,  # Корневая директория с изображениями
    batch_size: int = BATCH_SIZE,
    epochs: int = 10,
    learning_rate: float = 0.001,
    wandb_project: str = "custom-dataset-training-and-benchmark",
):
    """
    Запуск обучения модели.
    """
    logger.info("Initializing training...")

    # === Логирование W&B ===
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

    wandb.init(project=wandb_project, name=f"custom_model_run-{timestamp}")

    # === Подготовка данных ===
    transform_train = transforms.Compose([
        RandomResizedCrop(32),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



    train_dataset = SafeDataset(train_csv, root_dir, transform=transform_train)
    test_dataset = SafeDataset(test_csv, ROOT_DIR, transform=transform_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # === Подготовка модели ===
    model = create_custom_model(len(train_dataset.classes))
    model = model.to(DEVICE)

    # === Оптимизатор и функция потерь ===
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # === Обучение ===
    train_model(model, train_loader, optimizer, criterion, DEVICE, epochs, batch_size, learning_rate)

    # Оценка модели
    logger.info("Evaluating model performance...")
    custom_accuracy, custom_report = evaluate_model(model, test_loader, train_dataset.classes, DEVICE)
    logger.info(f"Model accuracy: {custom_accuracy:.2f}%")
    logger.info(f"Classification report:\n{custom_report}")
    wandb.log({
        "accuracy": custom_accuracy,
        "classification_report": custom_report,
        "model": "Custom ResNet18",
    })

    # === Сохранение модели ===
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model training complete. Model saved to {model_path}.")

    # === Завершение W&B ===
    wandb.finish()


if __name__ == "__main__":
    app()
