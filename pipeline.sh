#!/bin/bash

# Имя папки
FOLDER_NAME="$1"

# Добавляем суффикс -rescaled
MODIFIED_FOLDER_NAME="${FOLDER_NAME}-rescaled"

# Запуск первого Python-скрипта
python3.12 resize.py $FOLDER_NAME 500,623

# Запуск второго Python-скрипта
echo "$MODIFIED_FOLDER_NAME" | python3.12 back.py

rm -r $MODIFIED_FOLDER_NAME
