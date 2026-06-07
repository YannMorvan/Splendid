import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Classes in label order (extend with 'concealed' when that data is added)
CLASS_NAMES = ['with_mask', 'without_mask']
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FaceMaskDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples   # list of (path, int_label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def _collect_samples(data_dir):
    samples = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Expected class folder: {cls_dir}")
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples


def _transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_dir, batch_size=32, seed=42, num_workers=2):
    samples = _collect_samples(data_dir)
    labels  = [s[1] for s in samples]

    # Stratified 70/15/15 split
    train_s, temp_s, _, temp_l = train_test_split(
        samples, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    val_s, test_s = train_test_split(
        temp_s, test_size=0.50, stratify=temp_l, random_state=seed
    )

    train_tf, eval_tf = _transforms()

    train_dl = DataLoader(
        FaceMaskDataset(train_s, train_tf),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        FaceMaskDataset(val_s, eval_tf),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_dl = DataLoader(
        FaceMaskDataset(test_s, eval_tf),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Dataset split — train: {len(train_s)}  val: {len(val_s)}  test: {len(test_s)}")
    return train_dl, val_dl, test_dl


def get_class_weights(data_dir, device):
    """Inverse-frequency class weights for focal loss."""
    samples = _collect_samples(data_dir)
    counts = [0] * NUM_CLASSES
    for _, lbl in samples:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / (NUM_CLASSES * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float, device=device)
