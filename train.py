import torch
import wandb

def train_model(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            if inputs.shape[0] == 0:  # Пропускаем пустые данные
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})