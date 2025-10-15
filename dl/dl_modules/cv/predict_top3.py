

def predict_top3(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Добавляем размерность батча
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    # Сопоставляем индексы с метками
    results = []
    for i in range(3):
        label = classes[top3_indices[i].item()]
        prob = top3_probs[i].item() * 100  # Представление в процентах
        results.append(f"{label}: {prob:.2f}%")
    
    return results 