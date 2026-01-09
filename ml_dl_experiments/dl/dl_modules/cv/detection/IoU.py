import torch

Tensor = torch.Tensor

def calculate_iou(box1: Tensor, box2: Tensor, eps: float=1e-7) -> Tensor:
    """
    Вычисляет метрику Intersection over Union (IoU) для двух ограничивающих рамок.

    IoU определяется как отношение площади пересечения двух прямоугольников
    к площади их объединения и широко используется в задачах компьютерного
    зрения (детекция объектов, трекинг и т.д.).

    Формат входных рамок: [x1, y1, x2, y2], где:
        - (x1, y1) — координаты левого верхнего угла,
        - (x2, y2) — координаты правого нижнего угла.

    Args:
        box1 (Tensor): Тензор формы (4,), описывающий первую рамку.
        box2 (Tensor): Тензор формы (4,), описывающий вторую рамку.
        eps (float, optional): Малое значение для предотвращения деления на ноль.
            По умолчанию 1e-7.

    Returns:
        Tensor: Скалярный тензор, содержащий значение IoU в диапазоне [0, 1].
                Если площадь объединения равна нулю, возвращается 0.

    Example:
        >>> box1 = torch.tensor([0., 0., 2., 2.])
        >>> box2 = torch.tensor([1., 1., 3., 3.])
        >>> calculate_iou(box1, box2)
        tensor(0.1429)
    """

    # Координаты пересечения
    x1_inter: Tensor = torch.max(box1[0], box2[0])
    y1_inter: Tensor = torch.max(box1[1], box2[1])
    x2_inter: Tensor = torch.min(box1[2], box2[2])
    y2_inter: Tensor = torch.min(box1[3], box2[3])
    
    # Площадь пересечения
    inter_area: Tensor = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    
    # Площади исходных рамок
    box1_area: Tensor = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area: Tensor = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Площадь объединения
    union_area: Tensor = box1_area + box2_area - inter_area + eps
    
    # IoU
    iou: Tensor = inter_area / union_area if union_area > 0 else torch.tensor(0.0)
    return iou