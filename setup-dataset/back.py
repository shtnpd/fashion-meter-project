from rembg import remove
from PIL import Image
import os
import io  # Импортируем модуль io
from tqdm import tqdm

# Запрос имени входной папки у пользователя
input_folder = input("Введите путь к папке с изображениями: ").strip()

# Создание выходной папки
output_folder = f"{os.path.basename(input_folder)}-clean"
os.makedirs(output_folder, exist_ok=True)

# Список файлов в папке
file_list = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Обработка изображений с прогресс-баром
for filename in tqdm(file_list, desc="Обработка изображений"):
    filepath = os.path.join(input_folder, filename)

    try:
        # Открываем изображение
        with open(filepath, "rb") as input_file:
            image = input_file.read()

        # Удаляем фон
        output = remove(image)

        # Сохраняем результат
        output_image = Image.open(io.BytesIO(output))
        output_image = output_image.convert("RGB")  # Преобразование в RGB для сохранения в JPEG
        output_path = os.path.join(output_folder, filename)
        output_image.save(output_path)

    except Exception as e:
        print(f"Ошибка при обработке файла {filename}: {e}")

print(f"Обработка завершена. Все изображения сохранены в папке: {output_folder}")
