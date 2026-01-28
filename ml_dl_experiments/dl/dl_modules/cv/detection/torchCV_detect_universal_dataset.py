import os
from PIL import Image
import numpy as np
import torch
import torchvision

class PennFudanDataset(torch.utils.data.Dataset):
    """Решение для PennFudanDataset
    На его основе можно крафтить разные версии датасетов для детекции.
    
    Если мы поизучаем папку датасета, то найдем и информацию о bbox внутри папки Annotation в формате Pascal VOC. Можно добавить в этот класс реализацию по загрузки и этого решения.
    """
    def __init__(self, root, transforms=None):
        """
        Инициализатор класса.
        
        Args:
            root (str): Путь к корневой папке датасета ('PennFudanPed').
            transforms: Пайплайн трансформаций, которые будут применяться к данным.
        """
        self.root = root
        self.transforms = transforms
        # Загружаем все пути к изображениям и маскам.
        # Отсортируем их, чтобы img[i] соответствовал mask[i].
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, "PedMasks"))))

    def __len__(self):
        """Возвращает общее количество сэмплов в датасете."""
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Возвращает один сэмпл (изображение и таргет) по индексу.
        Это сердце нашего датасета, здесь происходит вся магия конвертации.
        """
        # Загружаем изображение и маску по их путям
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        # Открываем изображение в формате RGB. Это важно, так как модели ожидают 3 канала
        img = Image.open(img_path).convert("RGB")
        # Маску открываем как есть. У нее один канал
        mask = Image.open(mask_path)
        
        # Конвертируем маску в bounding box'ы
        # Маска - это изображение, где разные объекты размечены
        # разными цветами (разными целочисленными значениями пикселей).
        # Пиксели фона имеют значение 0.
        
        # Конвертируем маску PIL в массив numpy для удобной работы
        mask = np.array(mask)
        
        # Находим все уникальные ID объектов на маске.
        # Например, [0, 1, 2, 5], где 0 - фон, а 1, 2, 5 - id трёх разных пешеходов
        obj_ids = np.unique(mask)
        
        # Первое значение - это фон (0), поэтому мы его убираем.
        obj_ids = obj_ids[1:] # -> [1, 2, 5]

        # Теперь для каждого id объекта нужно создать отдельную бинарную маску
        # и найти по ней координаты bounding box
        binary_masks = (mask == obj_ids[:, None, None])

        # Получаем bounding box'ы из бинарных масок
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # Находим все координаты (y, x), где значение в маске равно True
            pos = np.where(binary_masks[i])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])

        # Собираем словарь target в требуемом формате
        
        # Конвертируем список с боксами в torch.Tensor типа float32
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # В этом датасете у нас только один класс - пешеход
        # Поэтому всем объектам присваиваем метку 1
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # Добавляем остальные поля, которые могут понадобиться
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Предположим, что все объекты не являются толпой
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Собираем финальный словарь target
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 4. Применяем трансформации (если они есть)
        if self.transforms is not None:
             img = torchvision.transforms.functional.to_tensor(img)  # type:ignore
        else:
             # Если трансформаций нет, всё равно нужно конвертировать img в тензор
             img = torchvision.transforms.functional.to_tensor(img)  # type:ignore


        return img, target