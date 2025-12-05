import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Функция для тренировки модели с выводом статистик
def train_image_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    num_epochs: int = 30,
    target_accuracy: float = 0.75,
    checkpoint_dir: str = "checkpoints",
    final_model_path: str = "meds_classifier.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Обучает модель CNN с логированием, чекпоинтами и ранней остановкой по точности.

    Args:
        model: Обучаемая модель 
        train_loader: DataLoader для тренировки (должен возвращать тензоры image, label ) 
        valid_loader: DataLoader для валидации (должен возвращать тензоры image, label )
        optimizer: Оптимизатор
        scheduler: Планировщик LR
        criterion: Функция потерь
        num_epochs: Макс. число эпох
        target_accuracy: Целевая точность (от 0.0 до 1.0)
        checkpoint_dir: Папка для чекпоинтов
        final_model_path: Путь к финальной модели
        device: Устройство (cuda/cpu)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print('='*50)

        # --- Обучение ---
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        pbar = tqdm(train_loader, desc="Train", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu()
            train_preds.extend(preds.tolist())
            train_targets.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": loss.item()})

        train_acc = accuracy_score(train_targets, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        # --- Валидация ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Valid", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu()
                val_preds.extend(preds.tolist())
                val_targets.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        avg_val_loss = val_loss / len(valid_loader)

        # Обновление LR
        scheduler.step(avg_val_loss) # type:ignore

        # --- Вывод эпохи ---
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {avg_val_loss:.4f} | Valid Acc: {val_acc:.4f}")

        # --- Чекпоинт каждые 10 эпох ---
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # --- Ранняя остановка по точности ---
        if val_acc >= target_accuracy:
            print(f"\nДостигнута целевая точность {target_accuracy:.2%} на валидации!")
            break

    # --- Сохранение финальной модели ---
    torch.save(model.state_dict(), final_model_path)
    print(f"\n Модель сохранена как: {final_model_path}")