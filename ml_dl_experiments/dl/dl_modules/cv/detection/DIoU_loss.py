import torch

from ml_dl_experiments.dl.dl_modules\
    .cv.detection.IoU import calculate_iou

tens = torch.Tensor
def diou_loss(
    preds: tens, 
    targets: tens, 
    eps: float=1e-7) -> tens:
    """
    Calculate DIoU loss by L = 1 − IoU + R(B, Bgt), 
    tens = torch.Tensor
    
    input bbox with template = [center_x, center_y, w, h]

    Args:
        preds (tens): Tensor predicted bbox
        targets (tens): Tensor target bbox
        eps (float, optional): eps for stability. Defaults to 1e-7.

    Returns:
        tens: torch.Tensor - loss value
    """
    # Конвертируем рамки в формат [x1, y1, x2, y2]
    preds_x1: tens = preds[..., 0] - preds[..., 2] / 2
    preds_y1: tens = preds[..., 1] - preds[..., 3] / 2
    preds_x2: tens = preds[..., 0] + preds[..., 2] / 2
    preds_y2: tens = preds[..., 1] + preds[..., 3] / 2
    
    targets_x1: tens = targets[..., 0] - targets[..., 2] / 2
    targets_y1: tens = targets[..., 1] - targets[..., 3] / 2
    targets_x2: tens = targets[..., 0] + targets[..., 2] / 2
    targets_y2: tens = targets[..., 1] + targets[..., 3] / 2

    # # Вычисляем IoU
    # # Найдем координаты области пересечения
    # inter_x1: tens = torch.max(preds_x1, targets_x1)
    # inter_y1: tens = torch.max(preds_y1, targets_y1)
    # inter_x2: tens = torch.min(preds_x2, targets_x2)
    # inter_y2: tens = torch.min(preds_y2, targets_y2)

    # # Площадь пересечения. Используйте clamp для избежания отрицательных значений.
    # inter_area: tens = torch.clamp(inter_x2 - inter_x1, min=0)\
    #     * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # # Площади предсказанных и истинных рамок
    # preds_area: tens = preds[..., 2] * preds[..., 3]
    # targets_area: tens = targets[..., 2] * targets[..., 3]
    
    # # Площадь объединения
    # union_area: tens = preds_area + targets_area - inter_area + eps
    
    # # Итоговый IoU
    # iou: tens = inter_area / union_area
    iou: tens = calculate_iou(
        torch.stack([preds_x1, preds_y1, preds_x2, preds_y2]),
        torch.stack([targets_x1, targets_y1, targets_x2, targets_y2]),
        eps=eps)

    # Вычисляем штраф за расстояние (Distance Penalty)
    # Квадрат расстояния между центрами рамок (rho^2)
    center_dist_sq: tens = (preds[..., 0] - targets[..., 0])**2 + (preds[..., 1] - targets[..., 1])**2

    # Найдём координаты углов наименьшей рамки, которая охватывает обе (enclosing box)
    enclose_x1: tens = torch.min(preds_x1, targets_x1)
    enclose_y1: tens = torch.min(preds_y1, targets_y1)
    enclose_x2: tens = torch.max(preds_x2, targets_x2)
    enclose_y2: tens = torch.max(preds_y2, targets_y2)

    # Квадрат диагонали охватывающей рамки (c^2). Добавляем eps для стабильности
    enclose_diag_sq: tens = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + eps

    # Штраф за расстояние (отношение квадрата расстояния между центрами к квадрату диагонали охватывающей рамки)
    distance_penalty: tens = center_dist_sq / enclose_diag_sq

    # Собираем всё вместе
    diou: tens = iou - distance_penalty
    
    # DIoU Loss определяется как 1 - DIoU
    loss: tens = 1.0 - diou
    return loss