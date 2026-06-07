import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from src.dataset import CLASS_NAMES


def evaluate(model, loader, device):
    """Return macro-F1 and collect all predictions."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1, np.array(all_preds), np.array(all_labels)


def full_report(model, loader, device, save_path=None):
    """Print classification report and plot confusion matrix."""
    _, preds, labels = evaluate(model, loader, device)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    out = save_path or 'confusion_matrix.png'
    plt.savefig(out, dpi=150)
    print(f"Confusion matrix saved → {out}")
    plt.close()

    macro_f1 = f1_score(labels, preds, average='macro')
    print(f"\nMacro-F1: {macro_f1:.4f}")
    return macro_f1
