import time
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from tqdm import tqdm
import typer

from fashion_meter_module.config import MODELS_DIR, PROCESSED_DATA_DIR, DEVICE, TEST_CSV, EPOCHS, LEARNING_RATE
from fashion_meter_module.evaluate import evaluate_model
from fashion_meter_module.model import create_custom_model
from fashion_meter_module.dataset import collate_fn, SafeDataset, get_transforms, get_dataloaders
from fashion_meter_module.config import TRAIN_CSV, ROOT_DIR, BATCH_SIZE
from torch.utils.data import DataLoader
from torchvision import transforms

app = typer.Typer()

import wandb
from torchvision.transforms import ColorJitter, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize

def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs, batch_size, learning_rate):
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

        # Оценка модели на тестовом наборе
        accuracy, report = evaluate_model(model, test_loader, test_loader.dataset.classes, device)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {accuracy:.2f}%")
        logger.info(f"Classification Report:\n{report}")

        # Логирование в WandB
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "test_accuracy": accuracy,
            "classification_report": report,
        })

    # wandb.finish()


@app.command()
def main(
    model_path: Path = MODELS_DIR / f"custom_model_resnet18_{time.time_ns()}.pth",
    train_csv: Path = TRAIN_CSV,
    test_csv: Path = TEST_CSV,
    root_dir: Path = ROOT_DIR,  # Корневая директория с изображениями
    batch_size: int = BATCH_SIZE, # 64
    epochs: int = EPOCHS, # 10
    learning_rate: float = LEARNING_RATE,
    wandb_project: str = "custom-dataset-training-and-benchmark",
    aug_mode: str = Literal['basic', 'basic+', 'best']
):
    """
    Запуск обучения модели.
    """
    logger.info("Initializing training...")

    # === Логирование W&B ===
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

    wandb.init(project=wandb_project, name=f"custom_model_run-{timestamp}")

    # === Подготовка данных ===
    transform_train, transform_test = get_transforms(aug_mode)

    train_dataset = SafeDataset(train_csv, root_dir, transform=transform_train)
    test_dataset = SafeDataset(test_csv, ROOT_DIR, transform=transform_test)

    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size)

    # === Подготовка модели ===
    model = create_custom_model(len(train_dataset.classes))
    model = model.to(DEVICE)

    # === Оптимизатор и функция потерь ===
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # === Обучение ===
    train_model(model, train_loader, test_loader, optimizer, criterion, DEVICE, epochs, batch_size, learning_rate)

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
