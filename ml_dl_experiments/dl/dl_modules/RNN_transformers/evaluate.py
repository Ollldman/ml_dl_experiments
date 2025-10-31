from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm

# код подсчёта accuracy на валидации
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    print("Start calculate accuracy")
    with torch.no_grad():
        for batch in tqdm(loader, ascii=True):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label']
            logits = model(ids, mask)
            preds += torch.argmax(logits, dim=1).cpu().tolist()
            trues += labels.tolist()
    return accuracy_score(trues, preds)