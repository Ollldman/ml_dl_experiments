from PIL import Image, ImageFilter
import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import requests
import matplotlib.pyplot as plt
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


image = Image.open("/data/sneakers.jpg").convert('RGB')

# Добавляем размытие с помощью фильтра GaussianBlur
blurred_image =image.filter(ImageFilter.GaussianBlur(radius=10))

# Сравниваем предсказания для оригинального изображения и изображения с шумом
original_pred = predict_top3(image)
blurred_pred = predict_top3(blurred_image)

print("Оригинал:", original_pred)
print("С шумом:", blurred_pred)

# Отображаем оригинальное и размытое изображение рядом
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Оригинальное изображение")
axes[0].axis("off")

axes[1].imshow(blurred_image)
axes[1].set_title("Изображение с GaussianBlur (radius=10)")
axes[1].axis("off")

plt.suptitle("Сравнение исходного изображения и изображения с шумом", fontsize=16)
plt.tight_layout()
plt.show()