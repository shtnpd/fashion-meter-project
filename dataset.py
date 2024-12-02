import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

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
        folder_with_suffix = f"{folder}-clean-rescaled"  # Добавляем суффикс
        img_path = os.path.join(self.root_dir, folder_with_suffix, filename)

        if not os.path.exists(img_path):
            print(f"Warning: Файл {img_path} не найден. Пропуск.")
            return None  # Пропускаем отсутствующий файл

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[folder]
        return image, label

def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Убираем None из батча
    if len(batch) == 0:  # Если весь батч пустой, возвращаем пустые тензоры
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)