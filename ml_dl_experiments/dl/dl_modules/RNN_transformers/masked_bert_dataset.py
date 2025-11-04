from torch.utils.data import Dataset
import torch

# класс датасета
class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        # self.samples - список пар (x, y)
        # x - токенизированный текст с пропущенным токеном
        # y - пропущенный токен
        self.samples = []

        for line in texts:
            token_ids = tokenizer.encode(
                line,
                add_special_tokens=False,
                truncation=True,
                max_length=512,
                return_tensors='pt').squeeze()
            # Проверка на пустую последовательность
            if token_ids.numel() == 0:
                continue
                
            # Если это тензор с одним элементом, преобразуем правильно
            if token_ids.dim() == 0:
                token_ids = token_ids.unsqueeze(0)
            
            token_ids = token_ids.tolist()
            # если строка слишком короткая, то пропускаем её
            if len(token_ids) < seq_len:
                continue
            
            
            # проходимся по всем токенам в последовательности
            for i in range(1, len(token_ids) - 1):
                '''
                context - список из seq_len // 2 токенов до i-го токена, токена tokenizer.mask_token_id, и seq_len // 2 токенов после i-го токена
                '''
                context = token_ids[max(0, i - seq_len//2): i] + [tokenizer.mask_token_id] + token_ids[i+1: i+1+seq_len//2]
                # если контекст слишком короткий, то пропускаем его
                if len(context) < seq_len:
                    continue
                target = token_ids[i]
                self.samples.append((context, target))
           
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)