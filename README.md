# Fashion Meter App

by Polina Ishutina and Dmitriy Ramus

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

![image](https://github.com/user-attachments/assets/92579023-d94a-4679-b9ef-b08bd9d40339)
Welcome to Fashion Meter, a project dedicated to making AI as stylish as your wardrobe! 👗👚👠

This project is an exploration of machine learning in the world of fashion, where we teach a neural network to classify fashion images into distinct styles. Whether you're a Cottagecore enthusiast, a fan of Y2K, or you just love some Academia chic, Fashion Meter can help you identify your next outfit trend with just a glance.

Fashion Meter uses cutting-edge Deep Learning models to analyze fashion images, and based on their content, it predicts which style category the image belongs to.

Here’s how it works:

**The Workflow**:
-- Data Preprocessing: We start with a raw dataset of fashion images (because raw fashion is the best, right?). The images are carefully organized into categories like Academia, Alt, Cottagecore, and Y2K.
-- Model Training: We feed these images into our custom-trained deep learning model, where it learns to recognize different fashion styles. Each epoch takes the model one step closer to understanding what makes each style unique.
-- Evaluation: After training, we evaluate the model’s performance and check how well it predicts which category an image belongs to. Spoiler: It does an amazing job at it.
-- Fashion Predictions: With a trained model in hand, you can now upload your favorite fashion images and get them classified into their rightful fashion styles!

**Why You Should Care**:
-- Perfect for fashion enthusiasts: Want to know which style is "hot" this season? This model can help!
-- Great for data science learners: Whether you're just starting out with machine learning or you're deep into the world of neural networks, this project has something to offer.
-- Fun & Practical: Because who doesn't love a bit of fashion AI fun?

**Features:**
-- Custom dataset of fashion images sorted by style.
-- Powerful neural network model that learns to classify images into fashion categories.
-- Integration with Google Colab and WandB for smooth experimentation and visualization.
--Easy-to-understand code and notebooks, ready for you to clone and run on your own.


So, why not give it a try? Dive into the world of Fashion Meter and let’s see if AI can predict the next big trend before anyone else does! 🚀

## Steps

Preprocess dataset before training via command:

### Prepare dataset

```shell
python data/raw/test-train.py
```

```sh

python -m fashion_meter_module.dataset process-raw-dataset \
    --input-csv data/raw/train.csv \
    --output-csv data/processed/train.csv
```
```sh

python -m fashion_meter_module.dataset process-raw-dataset \
    --input-csv data/raw/test.csv \
    --output-csv data/processed/test.csv
```

### Start training

```sh
python -m fashion_meter_module.modeling.train \
    --model-path models/custom_model_resnet18.pth \
    --train-csv data/processed/train.csv \
    --root-dir data/processed \
    --batch-size 64 \
    --epochs 15 \
    --learning-rate 0.0005 \
    --wandb-project "resnet18-custom-dataset"
```

### Use benchmarks

```sh
python -m fashion_meter_module.modeling.benchmark 
    --test-csv data/processed/test.csv 
    --root-dir data/processed 
    --batch-size 64 
    --models resnet18,resnet50,resnet101
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fashion_meter_module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── fashion_meter_module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fashion_meter_module a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

