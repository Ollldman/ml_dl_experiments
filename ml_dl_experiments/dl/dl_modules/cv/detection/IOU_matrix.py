from torch import (
    max,
    min,
    tensor,
    Tensor
)


def calculate_iou_matrix(
    boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculate IOU

    Args:
        boxes1 (Tensor): prediction boxes shape[0] > 1
        boxes2 (Tensor): ground trues boxes shape[0] > 1
        eps (float, optional): Non-zero deviding exp. Defaults to 1e-7.

    Returns:
        Tensor: shape[0] = (prediction.shape[0], ground trues.shape[0])
    """
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Вычислите площади рамок
    area1: Tensor = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2: Tensor = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Координаты пересечения
    inter_x1: Tensor = max(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
    inter_y1: Tensor = max(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
    inter_x2: Tensor = min(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
    inter_y2: Tensor = min(y2_1.unsqueeze(1), y2_2.unsqueeze(0))

    # Площадь пересечения
    inter_area: Tensor = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Площадь объединения
    union_area: Tensor = (area1.unsqueeze(1) + area2.unsqueeze(0)) - inter_area + tensor(eps)

    
    # Добавляем маленький эпсилон для избежания деления на ноль
    iou: Tensor = inter_area / union_area
    return iou