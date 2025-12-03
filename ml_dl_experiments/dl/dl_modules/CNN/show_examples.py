import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def show_examples(
        data: Dataset,
        num_examples: int,
        classes: list[str],
        denormalize: bool = True) -> None:
    rows: int = 1 if num_examples < 10 else (num_examples + 9) // 10

    fig = plt.figure(figsize=(15+int(0.2*num_examples), 5+int(0.2*num_examples)))
    
    for index in range(1, num_examples+1):
        image, label = data[index - 1]
        if denormalize:
            img = image.clone()
            img = img * 0.5 + 0.5
        else:
            img = image

        img = img.permute(1, 2, 0)
        plt.subplot(rows, 10, index)
        plt.imshow(img.numpy())
        plt.axis('off')
        plt.title(classes[label])

    plt.suptitle("Examples:")
    plt.tight_layout()
    plt.show()