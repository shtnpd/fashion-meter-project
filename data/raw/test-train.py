import os
import random
import csv


# Функция для разделения файлов на train и test
def split_files(folder_path, train_ratio=0.8):
    train_data = []
    test_data = []

    # Получаем список всех подпапок
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)

        # Обрабатываем название папки
        folder_label = folder_name.split('-')[0]  # Используем label без суффиксов

        # Проверяем, является ли элемент папкой
        if os.path.isdir(folder_full_path):
            try:
                files = os.listdir(folder_full_path)  # Читаем файлы в папке
                print(f"В папке {folder_name} {len(files)} файлов")
                random.shuffle(files)  # Перемешиваем файлы

                # Вычисляем количество файлов для train
                split_index = int(len(files) * train_ratio)
                train_files = files[:split_index]
                test_files = files[split_index:]
                print(f"Train: {len(train_files)} файлов")
                print(f"Test: {len(test_files)} файлов")

                # Формируем записи для CSV с относительными путями
                train_data.extend([(os.path.join(folder_name, file), folder_label) for file in train_files])
                test_data.extend([(os.path.join(folder_name, file), folder_label) for file in test_files])
            except PermissionError:
                print(f"Отказано в доступе к папке: {folder_full_path}")
            except Exception as e:
                print(f"Произошла ошибка при обработке папки {folder_full_path}: {e}")

    return train_data, test_data


# Функция для сохранения данных в CSV
def save_to_csv(data, output_path):
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "folder"])  # Заголовки
        writer.writerows(data)


# Главная функция
def main():
    base_folder_path = "./data/raw"  # Папка с исходными данными
    train_csv_path = "./data/processed/train.csv"
    test_csv_path = "./data/processed/test.csv"

    train_data, test_data = split_files(base_folder_path)

    save_to_csv(train_data, train_csv_path)
    save_to_csv(test_data, test_csv_path)

    print(f"Файлы train.csv и test.csv успешно созданы!")


if __name__ == "__main__":
    main()