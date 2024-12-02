import torch
from torch import nn
from torchvision import models
from fashion_meter_module.evaluate import evaluate_model
import typer
from pathlib import Path
from fashion_meter_module.dataset import SafeDataset, collate_fn
from torch.utils.data import DataLoader

def benchmark_models(models_to_test, test_loader, dataset_classes, device):
    """
    Бенчмаркинг нескольких моделей на тестовом наборе данных.
    """
    results = {}
    for model_name, model in models_to_test.items():
        # Адаптация модели к числу классов
        if "fc" in dir(model):  # Для ResNet
            model.fc = nn.Linear(model.fc.in_features, len(dataset_classes))
        elif "classifier" in dir(model):  # Для моделей с classifier
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset_classes))
        model = model.to(device)

        # Оценка модели
        accuracy, report = evaluate_model(model, test_loader, dataset_classes, device)
        print(f"Модель: {model_name}, Точность: {accuracy:.2f}%")
        print(report)
        results[model_name] = {"accuracy": accuracy, "report": report}
    return results

def load_models(model_names: str, dataset_classes: list, device: torch.device):
    """
    Load pretrained models specified by name and adapt them for the given number of classes.
    """
    model_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    models_to_test = {}
    for model_name in model_names.split(","):
        if model_name not in model_map:
            typer.secho(f"Model {model_name} is not supported.", fg=typer.colors.RED)
            continue
        model = model_map[model_name](pretrained=True)
        if "fc" in dir(model):  # Adjust the fully connected layer
            model.fc = nn.Linear(model.fc.in_features, len(dataset_classes))
        elif "classifier" in dir(model):  # For models with a classifier attribute
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset_classes))
        model = model.to(device)
        models_to_test[model_name] = model
    return models_to_test

app = typer.Typer()


@app.command()
def main(
    test_csv: Path = typer.Option(..., "--test-csv", help="Path to the test CSV file"),
    root_dir: Path = typer.Option(..., "--root-dir", help="Path to the root directory"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for testing"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", help="Device to use (cuda or cpu)"),
    model_names: str = typer.Option("resnet18,resnet50", "--models", help="Comma-separated list of model names to benchmark"),
):
    """
    Benchmark pretrained models on the provided dataset.
    """
    # Load dataset
    transform = SafeDataset.default_transform()  # Assuming you have a default transform defined
    test_dataset = SafeDataset(test_csv, root_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # Load models
    device = torch.device(device)
    dataset_classes = test_dataset.classes
    models_to_test = load_models(model_names, dataset_classes, device)

    # Benchmark models
    results = {}
    for model_name, model in models_to_test.items():
        accuracy, report = evaluate_model(model, test_loader, dataset_classes, device)
        typer.echo(f"Model: {model_name}, Accuracy: {accuracy:.2f}%")
        typer.echo(report)
        results[model_name] = {"accuracy": accuracy, "report": report}

    typer.secho("Benchmarking complete!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()