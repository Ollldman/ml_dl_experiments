import torch
from torch.utils.data import Dataset

# создаем датасет, наследуясь от класса Dataset из PyTorch
class RawDataset(Dataset):
    """
        возвращаем текст и его класс

        для текста ограничиваем длину
        
        не делаем никаких доп. преобразований как padding и masking
    """
    # в конструкторе просто сохраняем тексты и классы
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    # возвращаем размер датасета (кол-во текстов)
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
       
        return {
            'text': torch.tensor(self.texts[idx][:self.max_len], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }