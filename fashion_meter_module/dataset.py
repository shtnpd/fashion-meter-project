from pathlib import Path
import os
from typing import Literal

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
from torchvision.transforms.v2 import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, \
    Normalize
from tqdm import tqdm
import typer

from fashion_meter_module.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# === SafeDataset Class ===
class SafeDataset(Dataset):
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
        folder, filename = row['folder'], row['filename']

        # Формируем путь к изображению
        img_path = os.path.join(self.root_dir, filename)

        if not os.path.exists(img_path):
            logger.warning(f"Файл {img_path} не найден. Пропуск.")
            return None

        # Открываем и преобразуем изображение
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Получаем метку класса
        label = self.class_to_idx[folder]
        return image, label

    @staticmethod
    def default_transform():
        """
        Возвращает стандартные трансформации для изображений.
        """
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


# === Collate Function ===
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


# === Get Transforms ===
def get_transforms(mode: str = 'basic+'):
    logger.info(f"Аугментация - {mode}")
    if mode == 'basic+':
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

    elif mode == 'best':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),  # <-- Или тоже RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
    else:
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

    return transform_train, transform_test


# === Get DataLoaders ===
def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


# === CLI Commands ===
@app.command()
def check_dataset_integrity(
    input_csv: Path = RAW_DATA_DIR / "train.csv",
    root_dir: Path = RAW_DATA_DIR,
):
    """
    Проверка датасета на существование всех файлов.
    """
    logger.info(f"Проверка целостности датасета {input_csv}")
    missing_files = 0
    data = pd.read_csv(input_csv)
    for _, row in tqdm(data.iterrows(), total=len(data)):
        folder = f"{row['folder']}-clean-rescaled"
        filename = row["filename"]
        img_path = os.path.join(root_dir, folder, filename)
        if not os.path.exists(img_path):
            logger.warning(f"Файл отсутствует: {img_path}")
            missing_files += 1

    if missing_files == 0:
        logger.success("Все файлы на месте!")
    else:
        logger.error(f"Обнаружено {missing_files} отсутствующих файлов.")


@app.command()
def process_raw_dataset(
    input_csv: Path = RAW_DATA_DIR / "train.csv",
    output_csv: Path = PROCESSED_DATA_DIR / "train.csv",
):
    logger.info(f"Обработка датасета {input_csv}")
    data = pd.read_csv(input_csv)

    data["processed"] = True
    data.to_csv(output_csv, index=False)
    logger.success(f"Обработка завершена. Сохранено в {output_csv}")


if __name__ == "__main__":
    app()