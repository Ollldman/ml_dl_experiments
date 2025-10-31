import torch
from torch.utils.data import Dataset

class YelpDataset(Dataset):
    # в конструкторе просто сохраняем тексты и классы
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len


    # возвращаем размер датасета (кол-во текстов)
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        # возвращаем текст и его класс
        # для текста ограничиваем длину
        # не делаем никаких доп. преобразований как padding и masking
        text = self.texts[idx][:self.max_len]
        return {
            # верните текст под индексом idx в виде тензора, ограничьте его длиной self.max_len
            'text': \
                torch.tensor(text, dtype=torch.long),

            'label':\
                  torch.tensor(self.labels[idx], dtype=torch.long)
        }