# fashion_meter web app

by Dmitriy Ramus and Polina Ishutina

### Run locally without Docker
```shell
pip3 install -r requirements.txt
```

```shell
streamlit run main.py
```

### Run locally  with Docker

```shell
docker build -t fashion-meter-web-app .
```

```shell
docker run -p 8501:8501 fashion-meter-web-app
```
