import torch
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
# Загрузка меток классов
import requests
from predict_top3 import predict_top3

imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json()

# Загрузка предобученной модели ResNet и перевод её в режим инференса
# Создание пустой модели
model = models.resnet18(pretrained=True, weights="DEFAULT") 

# # Загрузка весов
# model.load_state_dict(torch.load('/models/resnet18_weights.pth'))
model.eval()

# Определение преобразований для изображения
transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


image = Image.open('/data/guitar.jpg').convert('RGB')
# Пример использования функции
predictions = predict_top3(model, image, classes)
print("Топ-3 предсказаний:")
for pred in predictions:
    print(pred)