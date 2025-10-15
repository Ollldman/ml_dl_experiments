import torch
from torchvision import models, transforms
from PIL import Image
# Загрузка меток классов
import requests
from predict_top3 import predict_top3

imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json()

# Загрузка предобученной модели ResNet и перевод её в режим инференса
# Создание пустой модели
model = models.resnet18(pretrained=False) 

# Загрузка весов
model.load_state_dict(torch.load('/models/resnet18_weights.pth'))
model.eval()

# Определение преобразований для изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


image = Image.open('/data/guitar.jpg').convert('RGB')
# Пример использования функции
predictions = predict_top3(image)
print("Топ-3 предсказаний:")
for pred in predictions:
    print(pred)