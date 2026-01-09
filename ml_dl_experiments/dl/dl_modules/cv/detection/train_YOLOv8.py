from ultralytics.models import YOLO
from pathlib import Path
from typing import Any

def train_yolov8(
    model_name: str, 
    yaml_path: Path,
    image_size: int,
    path_for_save: Path,
    num_epochs: int = 20,
    batch_size: int = 8,
    device: str = "cuda",
    patience: int = 10,
    **kwargs: Any) -> None:
    """
    Запускает обучение модели YOLOv8 на пользовательском датасете.

    Функция инициализирует предобученную модель YOLOv8 и выполняет её дообучение
    (fine-tuning) с использованием конфигурационного файла датасета в формате YAML.

    Параметры обучения (количество эпох, размер изображений, батч и т.д.)
    задаются напрямую внутри функции и могут быть изменены при необходимости.

    Args:
        model_name (str): Имя или путь к весам модели YOLOv8
            (например, "yolov8n.pt", "yolov8s.pt"). 
        yaml_path (Path): Путь к YAML-файлу с конфигурацией датасета
            (ключи: path, train, val, names и т.д.).
        path_for_save (Path): Путь или имя директории, в которую будут сохранены
            результаты эксперимента (логи, веса, графики).
        image_size (int): Типовой размер изображений,
        num_epochs (int): Число эпох для обучения Defaults to 20,
        batch_size (int): Размер батча. Defaults to 8
        device (str): device type, Defaults to "cuda"
        patience (int): Число эпох для ранней остановки, Defaults to 10
        **kwargs (Any): Дополнительные аргументы, передаваемые в `YOLO.train()`.
            Поддерживаются все параметры из официальной документации Ultralytics:
            `https://docs.ultralytics.com/modes/train/#apple-silicon-mps-training`
        
    Returns:
        None.  
        Информация о завершении обучения и путь к директории с результатами
        выводятся в стандартный поток вывода.

    Side Effects:
        - Создаёт директорию с результатами обучения.
        - Сохраняет веса модели (best.pt, last.pt).
        - Записывает логи и метрики обучения.

    Example:
        >>> from pathlib import Path
        >>> train_yolov8(
        ...     model_name="yolov8n.pt",
        ...     yaml_path=Path("datasets/config/config.yaml"),
        ...     path_for_save=Path("runs/exp1")
        ... )
    """
    model = YOLO(model_name) 

    results = model.train(
        data=str(yaml_path),   # Путь к нашему "паспорту" датасета
        epochs=num_epochs,                 # Количество эпох
        imgsz=image_size,                  # Размер изображений

        batch=batch_size,                   # Размер батча. Уменьшайте, если не хватает видеопамяти.
        device=device,                   # ID видеокарты (0) или 'cpu'.
        name=str(path_for_save), # Имя папки эксперимента.
        patience=patience,                # Количество эпох без улучшения для ранней остановки.
        **kwargs
    )
    if results is not None:
        print("Обучение завершено. Результаты сохранены в папке:", results.save_dir)
    else:
        print("Results is None")