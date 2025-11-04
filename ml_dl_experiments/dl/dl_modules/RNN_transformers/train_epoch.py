import torch
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    print("Start epoch learning!")
    for batch in tqdm(loader, ascii=True):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



def train_epoch_bidir(model, loader, optimizer, criterion, device):
    """
    :return: train_loss (total_loss / len(loader))
    """
    model.train()
    total_loss = 0
    total_batches = 0
    print("Start epoch training!")
    for x_batch, y_batch in tqdm(loader, ascii=True, desc="Training!"):
        x = x_batch.to(device)
        y = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
    return total_loss / total_batches if total_batches > 0 else float('inf')