import torch

from ml_dl_experiments.dl.dl_modules.\
    cv.detection import calculate_ap_for_class

Tensor = torch.Tensor

def calculate_map(
    predictions: Tensor, 
    ground_truths: Tensor, 
    num_classes: int, 
    iou_threshold: float = 0.5
) -> float:
    """
    Расчитываем meanAP по всему батчу изображений.
    В разрезе всего батча предсказаний считаем метрику AP для каждого класса.
    

    Args:
        predictions (list): Список предсказаний по каждому изображению в батче
        ground_truths (list): Список истинных рамок для каждого изображения в батче
        num_classes (int): Сколько классов искать
        iou_threshold (float, optional): Порог по метрике IoU. Defaults to 0.5.

    Returns:
        float: метрика mAP
    """
    # Собираем все вместе
    average_precisions: list[float] = []

    # Открываем цикл по каждому классу
    for c in range(num_classes):
        # Соберите все предсказания и истинные рамки для текущего класса 'c'
        # со всех изображений.
        all_preds_c = [] # будет elem = [img_id, x1, y1, x2, y2, conf, class_id]
        all_gts_c = [] # будет elem =  [img_id, x1, y1, x2, y2, class_id]
        # Заполняем все предсказания и истинные рамки для класса
        for i in range(len(predictions)):
            # Предсказания для класса c
            preds_in_image = predictions[i]
            if preds_in_image.numel() == 0:
                continue
            class_preds_mask = preds_in_image[:, 5] == c
            for p in preds_in_image[class_preds_mask]:
                # p = [x1, y1, x2, y2, conf, class_id] → берём как есть
                # Добавляем индекс изображения
                all_preds_c.append([i, *p[: 5].tolist()])
                
            gts_in_image = ground_truths[i]
            if gts_in_image.numel() == 0:
                continue
            class_gts_mask = gts_in_image[:, 4] == c 
            # предполагается, что GT: [x1, y1, x2, y2, class_id]
            # Будем делать g: [x1, y1, x2, y2, class_id, img_id]
            for g in gts_in_image[class_gts_mask]:
                all_gts_c.append([i, g[:4].tolist()])
        # Получим общее число истинных рамок для класса
        num_gts = len(all_gts_c)
        
        if num_gts == 0:
            continue
        # Если предсказаний для этого класса нет - ар = 0
        if not all_preds_c:
            average_precisions.append(0.0)
            continue
        # Превращаем наши сборники в тензоры
        preds_tensor = torch.tensor(all_preds_c)
        gts_tensor = torch.tensor(all_gts_c)
        
        aps: float = calculate_ap_for_class(preds_tensor, gts_tensor, iou_threshold)
        average_precisions.append(aps)
        
    if not average_precisions:
        return 0.0

    return sum(average_precisions) / len(average_precisions)