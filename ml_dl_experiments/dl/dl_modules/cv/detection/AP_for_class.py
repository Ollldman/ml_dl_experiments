import torch
from collections import defaultdict

from ml_dl_experiments.dl.dl_modules.\
    cv.detection.IoU import calculate_iou

Tensor = torch.Tensor

def calculate_ap_for_class(
    pred_boxes: Tensor, 
    true_boxes: Tensor, 
    iou_threshold: float = 0.5
) -> float:
    """
    Расчет метрики AP в контексте класса, внутри изображения, т.к. можно подать разметки предсказаний с разных изображений, следует предусмотреть наличие img_id внутри тензоров предсказания и истинных рамок.
    
    Args:
        pred_boxes (Tensor): elem = [img_id, x1, y1, x2, y2, conf, class_id]
        true_boxes (Tensor): elem = [img_id, x1, y1, x2, y2, class_id]
        iou_threshold (float, optional): Defaults to 0.5.

    Returns:
        float: AP метрика по классу, по img_id
    """
    # Если нет предсказаний, AP = 0.
    if pred_boxes.shape[0] == 0:
        return 0.0

    # Если нет истинных рамок, все предсказания - FP. AP = 0.
    if true_boxes.shape[0] == 0:
        return 0.0

    # Сортировка предсказаний по confidence
    sorted_indices: Tensor = torch.argsort(pred_boxes[:, 5], descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    # Массив для отслеживания TP/FP. 1 для TP, 0 для FP.
    tp: Tensor = torch.zeros(pred_boxes.shape[0])
    fp: Tensor = torch.zeros(pred_boxes.shape[0])
    
    gt_by_img = defaultdict(list)
    for i, gt in enumerate(true_boxes):
        img_id = int(gt[0].item())
        gt_by_img[img_id].append((i, gt[1:5]))
    # Флажки для использованных ground truth рамок
    # Для каждого изображения создаём флаги использованных GT
    gt_used = {img_id: torch.zeros(len(gts), dtype=torch.bool) for img_id, gts in gt_by_img.items()}

    for i, pred in enumerate(pred_boxes):
        img_id = int(pred[0].item())
        pred_box = pred[1:5]

        if img_id not in gt_by_img:
            fp[i] = 1
            continue

        gts = gt_by_img[img_id]
        used_flags = gt_used[img_id]

        # Считаем IoU только с GT из того же изображения
        ious = torch.tensor([
            calculate_iou(pred_box, gt_box) for _, gt_box in gts
        ])

        if ious.numel() == 0:
            fp[i] = 1
            continue

        best_iou, best_idx = ious.max(dim=0)

        if best_iou >= iou_threshold and not used_flags[best_idx]:
            tp[i] = 1
            used_flags[best_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / (true_boxes.shape[0] + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    recalls = torch.cat([torch.tensor([0.0]), recalls])
    precisions = torch.cat([torch.tensor([0.0]), precisions])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])

    idxs = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[idxs + 1] - recalls[idxs]) * precisions[idxs + 1])
    return ap.item()