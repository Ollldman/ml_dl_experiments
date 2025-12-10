import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Функция принимает на вход список элементов датасета, из них нужно сформировать батч.

    С помощью функции pad_sequence дополняет их до длины, равной длине максимальной последовательности в батче. В качестве padding_token используется 0.

    При сравнении токенов текстов после пэддинга с нулём (padding_token) считается маска.

    Возвращаемые значения  — тексты, маски, лейблы.
    """
    # список текстов и классов из батча
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    # дополняем тексты в батче padding'ом
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    # считаем маски
    masks = (padded_texts != 0).long()

    # возвращаем преобразованный батч
    return {
        'texts': padded_texts,
        'masks': masks,
        'labels': labels
    }

def collate_fn_with_sort(batch):
    
    sorted_batch = sorted(
        [(item["text"], item["label"]) for item in batch],
        key=lambda x: len(x[0]),
        reverse=True)
    texts = [i[0] for i in sorted_batch]
    labels = [i[1] for i in sorted_batch]
    lengths = [len(i[0]) for i in sorted_batch]
    # дополните тексты пэддингом
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    # посчитайте маску для батча
    masks = (padded_texts != 0).long()
    return {
        'texts': padded_texts,     # (batch_size, max_len)
        'masks': masks,            # (batch_size, max_len)
        'labels': labels,          # (batch_size,)
        "lengths": lengths
    }

def collate_fn_with_lengths(batch):
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([len(seq) for seq in texts], dtype=torch.long)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return {
        'input_ids': padded_texts, 
        'lengths': lengths, 
        'labels': labels
    }