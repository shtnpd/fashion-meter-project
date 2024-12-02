import torch
from torch import nn
from evaluate import evaluate_model

def benchmark_models(models_to_test, test_loader, dataset_classes, device):
    results = {}
    for model_name, model in models_to_test.items():
        if "fc" in dir(model):
            model.fc = nn.Linear(model.fc.in_features, len(dataset_classes))
        elif "classifier" in dir(model):
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset_classes))
        model = model.to(device)

        accuracy, report = evaluate_model(model, test_loader, dataset_classes, device)
        print(f"Модель: {model_name}, Точность: {accuracy:.2f}%")
        print(report)
        results[model_name] = {"accuracy": accuracy, "report": report}
    return results