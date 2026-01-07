import torch
from torch import Tensor

from ml_dl_experiments.dl.dl_modules.\
    cv.detection.IoU import calculate_iou

def non_max_suppression(
    predictions: Tensor, 
    conf_threshold: float=0.5, 
    iou_threshold: float=0.5) -> Tensor:
    """
    Реализует Non-Maximum Suppression, используя PyTorch.

    Args:
        predictions (torch.Tensor): Тензор предсказаний (N, 6) -> [x1, y1, x2, y2, conf, class_id].
        conf_threshold (float): Порог уверенности.
        iou_threshold (float): Порог IoU.
 
    Returns:
        torch.Tensor: Тензор отфильтрованных рамок (M, 6).
    """
    if predictions.shape[0] == 0:
        return torch.tensor([])
    # Фильтрация по уверенности (confidence score)
    preds: Tensor = predictions[predictions[:, 4] > conf_threshold]
    
    # Проверка на пустой результат
    # Если после фильтрации ничего не осталось:
    if preds.shape[0] == 0:
        return torch.tensor([])
    
    # Обработка предсказаний для каждого класса отдельно
    final_boxes: list[Tensor] = []
    # Найдите все уникальные ID классов в `preds`.
    unique_class_ids: Tensor = torch.unique(preds[:, 5])
    # Начните цикл по каждому уникальному ID класса.
    for class_id in unique_class_ids:
    # Внутри цикла по классам:
        # Выберите все предсказания, относящиеся к текущему классу.
        # Получаем тензор боксов для этого класса (Много рамок с близкими значениями)
        this_class_preds: Tensor = preds[preds[:, 5] == class_id]
        # Отсортируйте их по убыванию `confidence`.
        sorted_class_preds: Tensor =\
            this_class_preds[this_class_preds[:, 4].argsort(descending=True)]
        # начнем цикл проверки рамок по iou
        while sorted_class_preds.shape[0] > 0:
            # Берем рамку с лучшей уверенностью
            best_box: Tensor = sorted_class_preds[0]
            # Добавляем её в результат
            final_boxes.append(best_box)
            # Если это была последняя рамка, заканчиваем цикл
            if sorted_class_preds.shape[0] == 1:
                break
            # Убираем лучшую рамку из рассмотрения
            other_boxes: Tensor = sorted_class_preds[1:]
            
            # Рассчитываем IOU между лучшей рамкой с остальными
            """
                Это необходимо т.к. нам нужно для конкретного класса на изображении найти все релевантные рамки. (Ведь объект может быть представлен несколько раз на изображении в разных местах. В данном случае именно IOU threshold помогает нам отсеять дубликаты от настоящих "клонов" - других объектов с тем же классом, bboxs которых нам нужны)
                
                В таком случае, если мы оставляем рамки с наименьшим IOU мы именно гипотетически оставляем рамки на объекты расположенные в отдалении от того, который мы назначили "best_box" и они не являются дубликатами нашего best_box на тот же объект.
            """
            ious: Tensor = torch.tensor([calculate_iou(best_box[:4], box[:4]) for box in other_boxes])
            # Оставляем только рамки с наименьшим IOU
            sorted_class_preds = other_boxes[ious < iou_threshold]
            
    if not final_boxes:
        return torch.tensor([])

    return torch.stack(final_boxes)