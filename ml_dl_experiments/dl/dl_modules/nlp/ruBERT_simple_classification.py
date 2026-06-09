from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ml_dl_experiments.settings import settings

models_path: str = settings.SOURCE_PATH + "ml_dl/models"
# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(models_path + "/rubert-tiny-sentiment-balanced")
model = AutoModelForSequenceClassification.from_pretrained(models_path + "/rubert-tiny-sentiment-balanced")

# Перевод модели в режим оценки
model.eval()

print("Модель и токенизатор успешно загружены!")

# Пример текста на русском языке
text = "Эта гитара оставила у меня исключительно положительные впечатления!"

# Токенизация с использованием padding и truncation
inputs = tokenizer(
    text,
    padding='max_length',   # Все последовательности дополняются специальными токенами до заданной длины. Это необходимо, чтобы обеспечить одинаковый размер входа для всех примеров.
    truncation=True,        # Обрезка, если текст длиннее max_length
    max_length=64,          # Если текст превышает 64 токена, он обрезается до нужного размера, чтобы избежать ошибок при обработке.
    return_tensors="pt"     # Результат возвращается в формате тензоров PyTorch, что удобно для дальнейшей работы с моделью.
)

# Выполнение предсказания без вычисления градиентов
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Применяем softmax для получения вероятностей
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities).item()

# Выводим результат
print(f"\nПредсказанный класс: {predicted_class}")
print(f"Вероятности классов: {probabilities}")