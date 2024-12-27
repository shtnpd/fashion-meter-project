from pathlib import Path
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
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
def get_transforms():
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


# # === Get DataLoaders ===
# def get_dataloaders(train_csv, test_csv, root_dir, batch_size):
#     transform_train, transform_test = get_transforms()
#     train_dataset = SafeDataset(train_csv, root_dir, transform=transform_train)
#     test_dataset = SafeDataset(test_csv, root_dir, transform=transform_test)
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         collate_fn=collate_fn,
#     )
#
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn,
#     )
#
#     return train_loader, test_loader


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