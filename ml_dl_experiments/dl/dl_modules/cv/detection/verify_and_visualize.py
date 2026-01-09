import yaml
import cv2
from cv2.typing import MatLike
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt

def verify_and_visualize(data_yaml_path: str) -> None:
    """
    Читает data.yaml, выбирает случайное изображение из обучающей выборки
    и визуализирует его вместе с разметкой.
    path like:
    path_to_my_yaml = settings.SOURCE_PATH + "datasets/for_cv/config/config.yaml"
    Извлекаем из `data_config` следующие переменные:
    1. `class_names` (из ключа 'names')
    2. `train_images_rel_path` (из ключа 'train')
    3. `dataset_root_path_rel` (из ключа 'path')
    
    config.yaml looks like:
    
    ***
    path: ../transport_dataset
    
    train: images/train
    
    val: images/val
    
    nc: 3
    
    names: ['bike', 'bus', 'car']
    
    ***
    
    Directory looks like:
    
    ***
        datasets/
        ├──transport_dataset/
        │   ├── images/
        │   │   ├── train/  (содержит car1.jpg, bike1.jpg, ...)
        │   │   └── val/    (содержит car2.jpg, bike2.jpg, ...)
        │   └── labels/
        │       ├── train/  (содержит car1.txt, bike1.txt, ...)
        │       └── val/    (содержит car2.txt, bike2.txt, ...)
        │
        ├── config/
        │   └── config.yaml  // Конфиг
        │
        └── train.py       // Скрипт обучения
    ***
    """
    # Чтение и парсинг data.yaml
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        return

    class_names: list[str] = data_config['names']
    train_images_rel_path: str = data_config['train']
    dataset_root_path_rel: str = data_config['path']

    # Построение абсолютных путей
    yaml_path: Path  = Path(data_yaml_path).resolve()
    yaml_dir: Path = yaml_path.parent
    
    dataset_root_abs_path: Path = (yaml_dir / dataset_root_path_rel).resolve()

    
    train_images_abs_path: Path = (dataset_root_abs_path / train_images_rel_path).resolve()

    train_labels_abs_path: Path = Path(str(train_images_abs_path).replace("images", "labels")).resolve()


    # Выбор случайного изображения и его разметки
    all_images: list[Path] = [f for f in train_images_abs_path.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']]
    
    if not all_images:
        return
        
    random_image_name: Path = random.choice(all_images)
    image_path: Path = (train_images_abs_path / random_image_name).resolve()
    print("Image: ", image_path.name)
    label_name: Path = Path(random_image_name.name.replace(".jpg", ".txt"))
    label_path: Path = (train_labels_abs_path / label_name).resolve()
    print("Label: ", label_path.name)
    # Визуализация
    image: MatLike | None = cv2.imread(str(image_path))
    if image is not None:
        img_height, img_width, _ = image.shape
        
        if not label_path.exists():
            print("Для этого изображения нет файла разметки")
            
        else:
            with open(label_path, 'r') as f:
                lines: list[str] = f.readlines()
                if not lines:
                    print("Файл разметки пуст!")
                    return
                for line in lines:
                    parts: list[str] = line.strip().split()
                    print("bbox:\n", parts)
                    if len(parts) > 5:
                        print("Длина кодировка bbox больше 5!")
                        continue
                    # Распарсите строку из `.txt` файла.
                    # 1. Извлеките `class_id`, `x_center`, `y_center`, `width`, `height`.
                    # 2. Не забудьте преобразовать строки в `int` и `float`.
                    class_id: int = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Денормализация: Преобразуйте нормализованные координаты в пиксельные.
                    box_w: int = int(width * img_width)
                    box_h: int = int(height * img_height)
                    x_min: int = int((x_center * img_width) - box_w / 2)
                    y_min: int = int((y_center * img_height) - box_h / 2)
                    print("Scaled bbox:\n", [box_w, box_h, x_min, y_min])
                    
                    # Рисуем рамку и подпись
                    class_name: str = class_names[class_id]
                    cv2.rectangle(image, (x_min, y_min), (x_min + box_w, y_min + box_h), (0, 255, 0), 2)
                    cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Отображаем результат
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Верификация разметки: {random_image_name.name}")
        plt.axis('off')
        plt.show()
    else:
        print(f"Image {image_path} with label {label_path} is not exist!")
