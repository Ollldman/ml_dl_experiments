# 🎯 PyTorch Тензорные операции: руководство по размерностям

## 📐 Основные принципы размерностей

### Структура данных в CV:
```
Images:    (B, C, H, W)  # Batch, Channels, Height, Width
Masks:     (B, H, W) или (B, 1, H, W)
Features:  (B, C, H, W) → (B, C) после pooling
```

## 🔄 Основные операции изменения размерностей

### 1. **`permute()` - перестановка осей**
```python
# Из (B, C, H, W) в (B, H, W, C) для визуализации
img = torch.randn(4, 3, 256, 256)  # B, C, H, W
img_vis = img.permute(0, 2, 3, 1)   # B, H, W, C

# Для одного изображения:
img_single = img[0]                  # (C, H, W)
img_vis_single = img_single.permute(1, 2, 0)  # (H, W, C)
```

### 2. **`view()` - изменение формы (без копирования)**
```python
# Выравнивание в вектор
features = torch.randn(4, 512, 8, 8)  # (B, C, H, W)
flattened = features.view(4, -1)      # (B, 512*8*8) = (4, 32768)

# Из 1D в 4D
vector = torch.randn(4, 512)
reshaped = vector.view(4, 512, 1, 1)  # (B, C, 1, 1) для broadcast
```

### 3. **`reshape()` - изменение формы (с копированием если нужно)**
```python
# Безопаснее чем view()
tensor = torch.randn(4, 3, 256, 256)
reshaped = tensor.reshape(4, -1)  # (4, 196608)
back = reshaped.reshape(4, 3, 256, 256)
```

### 4. **`unsqueeze()` / `squeeze()` - добавление/удаление размерности**
```python
# Добавить batch dimension
img = torch.randn(3, 256, 256)      # (C, H, W)
img_batch = img.unsqueeze(0)        # (1, C, H, W)

# Добавить channel dimension для масок
mask = torch.randn(4, 256, 256)     # (B, H, W)
mask_channel = mask.unsqueeze(1)    # (B, 1, H, W)

# Удалить singleton dimensions
tensor = torch.randn(1, 3, 1, 256, 256)
clean = tensor.squeeze()            # (3, 256, 256) - удалит ВСЕ единичные
clean_safe = tensor.squeeze(dim=0)  # (3, 1, 256, 256) - только указанную
```

## 🖼️ Операции для работы с изображениями

### Конвертация для визуализации (matplotlib/OpenCV)
```python
import matplotlib.pyplot as plt
import torch
import numpy as np

def tensor_to_image(tensor):
    """
    Конвертирует тензор PyTorch в изображение для matplotlib
    Input: (C, H, W) или (B, C, H, W)
    Output: (H, W, C) numpy array
    """
    if tensor.dim() == 4:  # batch
        tensor = tensor[0]  # берем первый
    
    # Detach, move to CPU, convert to numpy
    img = tensor.detach().cpu()
    
    # Normalize если нужно
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    
    # Permute channels last
    if img.shape[0] == 3 or img.shape[0] == 1:  # CHW
        img = img.permute(1, 2, 0)
    
    return img.numpy()

# Использование
img_tensor = torch.randn(1, 3, 256, 256)  # модель выдает
img_np = tensor_to_image(img_tensor)      # (256, 256, 3)
plt.imshow(img_np)
plt.show()
```

### Нормализация/денормализация
```python
# ImageNet нормализация
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def normalize_imagenet(tensor):
    """Нормализация для предобученных моделей"""
    # tensor: (B, C, H, W) или (C, H, W)
    return (tensor - IMAGENET_MEAN.view(-1, 1, 1)) / IMAGENET_STD.view(-1, 1, 1)

def denormalize_imagenet(tensor):
    """Обратная нормализация для визуализации"""
    return tensor * IMAGENET_STD.view(-1, 1, 1) + IMAGENET_MEAN.view(-1, 1, 1)

# Пример
img = torch.rand(3, 256, 256)  # [0, 1]
normalized = normalize_imagenet(img)  # для модели
denormalized = denormalize_imagenet(normalized)  # для показа
```

## 🎯 Полезные паттерны для CV

### 1. **Broadcast операций**
```python
# Добавить канал к маске
mask = torch.randn(4, 256, 256)  # (B, H, W)
mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)

# Умножение mask на image
image = torch.randn(4, 3, 256, 256)
masked_image = image * mask_expanded  # broadcast: (B, 3, H, W) * (B, 1, H, W)
```

### 2. **Конкатенация по разным осям**
```python
# Concat по batch dimension
batch1 = torch.randn(2, 3, 256, 256)
batch2 = torch.randn(3, 3, 256, 256)
combined = torch.cat([batch1, batch2], dim=0)  # (5, 3, 256, 256)

# Concat по channel dimension
rgb = torch.randn(4, 3, 256, 256)
depth = torch.randn(4, 1, 256, 256)
rgbd = torch.cat([rgb, depth], dim=1)  # (4, 4, 256, 256)

# Stack для создания новой оси
tensors = [torch.randn(256, 256) for _ in range(5)]
stacked = torch.stack(tensors, dim=0)  # (5, 256, 256)
```

### 3. **Скользящее окно (patch extraction)**
```python
def extract_patches(tensor, patch_size=64, stride=32):
    """
    Извлекает патчи из изображения
    Input: (B, C, H, W)
    Output: (B * n_patches, C, patch_size, patch_size)
    """
    B, C, H, W = tensor.shape
    patches = tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, patch_size, patch_size)
    return patches

# Обратная операция
def reconstruct_from_patches(patches, original_shape, patch_size=64, stride=32):
    B, C, H, W = original_shape
    n_patches_h = (H - patch_size) // stride + 1
    n_patches_w = (W - patch_size) // stride + 1
    
    patches = patches.view(B, n_patches_h * n_patches_w, C, patch_size, patch_size)
    # ... реконструкция (сложнее, зависит от overlap)
```

## 📊 Работа с батчами

### 1. **Применение операции к каждому элементу батча**
```python
# Вариант 1: Vectorized (лучше)
def process_batch_vectorized(batch):
    """batch: (B, C, H, W)"""
    return batch * 2 + 1  # автоматический broadcast

# Вариант 2: Явный loop (иногда нужно)
def process_batch_loop(batch):
    results = []
    for i in range(batch.size(0)):
        img = batch[i]  # (C, H, W)
        processed = some_function(img)  # работает с 3D
        results.append(processed)
    return torch.stack(results, dim=0)
```

### 2. **Batch-wise statistics**
```python
# Среднее по батчу (но не по spatial dimensions)
batch = torch.randn(8, 3, 256, 256)
mean_per_image = batch.mean(dim=(2, 3))  # (8, 3) - mean per channel per image
mean_per_batch = batch.mean(dim=0)       # (3, 256, 256) - mean batch

# Normalize каждый image отдельно
def batch_instance_norm(batch):
    """Normalize каждый image в батче отдельно"""
    B, C, H, W = batch.shape
    batch_flat = batch.view(B, C, -1)  # (B, C, H*W)
    mean = batch_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
    std = batch_flat.std(dim=2, keepdim=True)    # (B, C, 1)
    normalized = (batch_flat - mean) / (std + 1e-5)
    return normalized.view(B, C, H, W)
```

## 🔧 Утилиты для отладки

```python
def print_tensor_info(tensor, name="Tensor"):
    """Печать информации о тензоре"""
    print(f"{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min():.4f}, Max: {tensor.max():.4f}")
    print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
    print(f"  Requires grad: {tensor.requires_grad}")
    if tensor.dim() == 4:
        print(f"  Format: (B={tensor.shape[0]}, C={tensor.shape[1]}, H={tensor.shape[2]}, W={tensor.shape[3]})")
    print()

def check_nan_inf(tensor):
    """Проверка на NaN/Inf значения"""
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  Warning: {nan_count} NaN, {inf_count} Inf values")
        return False
    return True
```

## 🚀 Быстрые преобразования (one-liners)

```python
# CHW → HWC для визуализации
vis = tensor.permute(1, 2, 0).cpu().numpy()

# Добавить batch dimension если нужно
if tensor.dim() == 3:
    tensor = tensor.unsqueeze(0)

# Удалить batch dimension если один элемент
if tensor.shape[0] == 1:
    tensor = tensor.squeeze(0)

# Конвертация bool маски в float
mask = (tensor > 0.5).float()

# One-hot encoding для семантической сегментации
def to_onehot(mask, num_classes):
    """mask: (B, H, W) или (H, W) с классами 0..num_classes-1"""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    B, H, W = mask.shape
    onehot = torch.zeros(B, num_classes, H, W, device=mask.device)
    onehot.scatter_(1, mask.long().unsqueeze(1), 1)
    return onehot

# Score to prediction
probs = torch.randn(4, 2, 256, 256)  # (B, C, H, W)
preds = probs.argmax(dim=1)  # (B, H, W) с классами 0 или 1
```

## 📝 Чекалист перед передачей в модель

```python
def prepare_for_model(tensor, model):
    """
    Подготовка тензора для модели
    """
    # 1. Правильная размерность
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # add batch dim
    
    # 2. Правильный dtype (обычно float32)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # 3. Правильный device
    if next(model.parameters()).device != tensor.device:
        tensor = tensor.to(next(model.parameters()).device)
    
    # 4. Нормализация если требуется
    if hasattr(model, 'mean') and hasattr(model, 'std'):
        tensor = (tensor - model.mean) / model.std
    
    # 5. Проверка значений
    assert not torch.isnan(tensor).any(), "NaN in input"
    
    return tensor
```

## 🎨 Пример полного пайплайна обработки

```python
def full_image_pipeline(image_path, model, device='cuda'):
    """
    Полный пайплайн: загрузка → обработка → модель → визуализация
    """
    # 1. Загрузка (OpenCV/PIL → numpy)
    import cv2
    img_np = cv2.imread(image_path)  # (H, W, 3) BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # RGB
    
    # 2. Preprocessing
    img_np = cv2.resize(img_np, (256, 256))  # resize
    img_tensor = torch.from_numpy(img_np).float() / 255.0  # [0, 1]
    
    # 3. CHW формат
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, 256, 256)
    
    # 4. Batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 256, 256)
    
    # 5. Normalize
    img_tensor = normalize_imagenet(img_tensor)
    
    # 6. Device
    img_tensor = img_tensor.to(device)
    
    # 7. Inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # 8. Postprocessing
    if output.dim() == 4 and output.shape[1] > 1:  # segmentation
        pred = output.argmax(dim=1)  # (1, H, W)
        pred = pred.squeeze(0)  # (H, W)
    else:
        pred = output
    
    # 9. Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(denormalize_imagenet(img_tensor[0]).permute(1, 2, 0).cpu())
    ax2.imshow(pred.cpu(), cmap='gray')
    
    return pred
```

## 💡 Главные правила

1. **Всегда проверяйте `.shape`** после сложных операций
2. **`.permute()` для изменения порядка осей**, не `view/reshape`
3. **`.unsqueeze(0)` для добавления batch dimension**
4. **`.squeeze()` для удаления единичных размерностей**
5. **`detach().cpu().numpy()` для конвертации в numpy**
6. **Нормализуйте изображения** как ожидает модель
7. **Используйте `.to(device)`** а не `.cuda()/.cpu()` напрямую
8. **Проверяйте NaN/Inf** в процессе отладки

---

**Запомните:** `(B, C, H, W)` → **B**atch, **C**hannels, **H**eight, **W**idth  
Для визуализации: `permute(0, 2, 3, 1)` или `permute(1, 2, 0)` для single image