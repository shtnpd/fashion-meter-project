# Dataset Setup Scripts

Эта папка содержит несколько скриптов и вспомогательных файлов для подготовки датасета для проекта, включая обработку изображений, работу с данными и запуск автоматизированных процессов. Ниже представлено описание структуры файлов.

## Содержимое

### Python-скрипты
- **back.py**  
  Скрипт отвечает за автоматическое удаление фона изображений датасета. Любой цветной фон заменяется на черный.

- **get-size.py**  
  Предназначен для расчета оптимального размера изображений по соотношению сторон. Помогает выбрать единое разрешение для всех изображений датасета.

- **pinterest.py**  
  Скрипт для парсинга изображений с платформы Pinterest по заданному промпту. Поможет найти изображения на платформе, если понадобится расширить датасет.

- **resize.py**  
  Скрипт для изменения размеров изображений. Полезен для подготовки данных и оптимизации ресурсов.

### Shell-скрипт
- **pipeline.sh**  
  Shell-скрипт, содержащий pipeline для централизованной обработки сырых данных. Включает все скрипты выше последовательно.

## Использование
1. Убедитесь, что у вас установлены все зависимости, необходимые для работы скриптов.
2. Для выполнения Python-скриптов используйте команду:
   ```bash
   python3 <название_скрипта>.py