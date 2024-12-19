import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Функция для создания модели
def create_custom_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_custom_model(num_classes, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

@st.cache_data
def load_labels(labels_path, num_classes):
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        if len(labels) != num_classes:
            st.warning(f"Количество меток ({len(labels)}) не совпадает с num_classes ({num_classes}). Используются стандартные имена классов.")
            labels = [f"Класс {i}" for i in range(num_classes)]
    else:
        st.warning("Файл labels.txt не найден. Используются стандартные имена классов.")
        labels = [f"Класс {i}" for i in range(num_classes)]
    return labels

MODEL_PATH = "custom_model.pth"
LABELS_PATH = "labels.txt"

num_classes = 4  # Например, 5 классов

model, device = load_model(MODEL_PATH, num_classes)
labels = load_labels(LABELS_PATH, num_classes)

st.title('Fashion Meter | Классификация одежды по фото')
st.write("""
Загрузите изображение, и модель выполнит его классификацию.
""")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Загруженное изображение.', use_container_width=True)

        st.write("### Предсказание модели:")
        with st.spinner('Классификация...'):
            # Преобразования, соответствующие модели
            preprocess = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # Создание мини-батча

            with torch.no_grad():
                input_batch = input_batch.to(device)
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Получение топ-k предсказаний
            topk_prob, topk_catid = torch.topk(probabilities, num_classes)
            for i in range(topk_prob.size(0)):
                class_id = topk_catid[i].item()
                if class_id < len(labels):
                    class_name = labels[class_id]
                else:
                    class_name = f"Класс {class_id}"
                st.write(f"{class_name}: {topk_prob[i].item() * 100:.2f}%")

    except Exception as e:
        st.error(f"Произошла ошибка при обработке изображения: {e}")
else:
    st.write("Пожалуйста, загрузите изображение для классификации.")


st.sidebar.header("Информация о модели")
st.sidebar.info(f"""
- **Тип модели:** ResNet18
- **Обучена на:** Pinterest dataset
- **Количество классов:** {num_classes}
""")

st.sidebar.header("Контакты")
st.sidebar.markdown("""
[GitHub](https://github.com/shtnpd/fashion-meter-project)
""")