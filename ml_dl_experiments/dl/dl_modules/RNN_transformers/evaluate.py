from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm

# код подсчёта accuracy на валидации
def evaluate_accuracy(model, loader, device):
    """
    return accuracy_score from sklearn.metrics
    """
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



def evaluate_bidir(model, loader, loss_func, device):
    """
    :return:
    avg_loss = summ_loss / len(loader);  accuracy = correct / total, 
    """
    model.eval()
    correct, total = 0, 0
    summ_loss = 0
    print("Start calculate scores")
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, ascii=True):
            x = x_batch.to(device)
            y = y_batch.to(device)
            outputs = model(x)
            loss = loss_func(outputs, y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            summ_loss += loss
            
    if total == 0:
        return float('inf'), 0.0
    
    avg_loss = summ_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy