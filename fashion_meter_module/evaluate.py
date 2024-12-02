import torch
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader, dataset_classes, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs.shape[0] == 0:  # Пропускаем пустые данные
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = 100 * correct / total
    report = classification_report(all_labels, all_preds, target_names=dataset_classes, zero_division=0)
    return accuracy, report