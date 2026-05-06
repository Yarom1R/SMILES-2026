import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True

def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    
    # 1. Загружаем полный тренировочный датасет (50 000 картинок)
    full_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    # 2. Группируем индексы по классам
    targets = np.array(full_dataset.targets)
    indices_per_class = [np.where(targets == i)[0] for i in range(100)]
    
    selected_indices = []
    
    # Чтобы выбор был воспроизводимым, используем локальный Random State
    # (или можно использовать generator_train, если нужно)
    rng = np.random.RandomState(42) 

    # 3. Берем по 81 картинке из каждого из 100 классов (итого 8100)
    for i in range(100):
        class_indices = indices_per_class[i]
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[:81].tolist())

    # 4. Нам нужно еще 92 картинки, чтобы добить до лимита 8192
    # Соберем все оставшиеся индексы, которые еще не использовали
    used_indices_set = set(selected_indices)
    remaining_indices = [i for i in range(len(targets)) if i not in used_indices_set]
    rng.shuffle(remaining_indices)
    
    selected_indices.extend(remaining_indices[:92])

    # 5. Создаем Subset из выбранных 8192 индексов
    train_subset = Subset(full_dataset, selected_indices)

    # 6. Создаем DataLoader
    # Важно: shuffle=True позволит перемешать эти отобранные классы внутри эпохи
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_subset, train_loader