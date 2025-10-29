from typing import List, Union, Dict, Any
import torch
from torch.utils.data import Dataset

class PaddedDataset(Dataset):
    """
    Dataset для текстов с ручным padding'ом и маскированием.

    Каждый элемент датасета — словарь с тремя ключами:
        - 'texts': токенизированный текст, дополненный до max_len нулями.
        - 'masks': бинарная маска (1 — реальный токен, 0 — padding).
        - 'labels': метка класса (скаляр или категориальный индекс).
    """

    def __init__(
        self,
        texts: List[List[int]],
        labels: List[Union[int, float]],
        max_len: int
    ) -> None:
        """
        Args:
            texts: Список токенизированных текстов, каждый — список целых чисел.
            labels: Список меток (обычно int для классификации).
            max_len: Максимальная длина последовательности после padding'а.
        """
        if len(texts) != len(labels):
            raise ValueError("Длины texts и labels должны совпадать.")
        if max_len <= 0:
            raise ValueError("max_len должен быть положительным целым числом.")
        self.texts: List[List[int]] = texts
        self.labels: List[Union[int, float]] = labels
        self.max_len: int = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        if len(text) > self.max_len:
            # Опционально: обрезка, если текст длиннее max_len
            text = text[:self.max_len]
            padded = text
            mask = [1] * len(text)
        else:
            padded = text + [0] * (self.max_len - len(text))
            mask = [1] * len(text) + [0] * (self.max_len - len(text))

        return {
            'texts': torch.tensor(padded, dtype=torch.long),
            'masks': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
        }