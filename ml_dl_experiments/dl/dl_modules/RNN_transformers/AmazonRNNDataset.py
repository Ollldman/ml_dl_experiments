from torch.utils.data import Dataset
import torch

class AmazonRNNDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': self.labels[idx]
        }