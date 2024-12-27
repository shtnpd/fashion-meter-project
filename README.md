# fashion_meter

by Dmitriy Ramus and Polina Ishutina

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

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

