from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

def predict_top3(model, image, classes, transform: Compose | None=None) -> list[str]:

    # Применяем преобразования
    if transform:
        image_tensor = transform(image).unsqueeze(0)  # Добавляем размерность батча: [1, C, H, W]
    else:
        transforming: Compose = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_tensor = transforming(image).unsqueeze(0)

    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)

    # Вероятности через softmax
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_probs, top3_indices = torch.topk(probabilities, 3)

    # Формируем результат
    results: list[str] = []
    for i in range(3):
        label = classes[top3_indices[i].item()]
        prob = top3_probs[i].item() * 100  # в процентах
        results.append(f"{label}: {prob:.2f}%")

    return results