import os
import torch
import torch.optim as optim
from tqdm import tqdm

from src.model import build_model, freeze_backbone, unfreeze_last_block
from src.focal_loss import FocalLoss
from src.dataset import get_dataloaders, get_class_weights, NUM_CLASSES
from src.evaluate import evaluate


DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset')
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')

# ── hyper-parameters (matching the proposal) ──────────────────────────────────
BATCH_SIZE   = 32
LR           = 3e-4
GAMMA_FOCAL  = 2.0
SEED         = 42

PHASE1_EPOCHS = 10   # head only
PHASE2_EPOCHS = 25   # unfreeze last block
PATIENCE      = 6    # early-stopping patience (on val macro-F1)
# ──────────────────────────────────────────────────────────────────────────────


def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for imgs, labels in tqdm(loader, leave=False, desc='train' if training else 'val'):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def train():
    device = _device()
    print(f"Using device: {device}")

    train_dl, val_dl, test_dl = get_dataloaders(DATA_DIR, BATCH_SIZE, SEED)

    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    class_weights = get_class_weights(DATA_DIR, device)
    criterion = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL)

    best_f1   = 0.0
    best_state = None

    # ── Phase 1: train head only ───────────────────────────────────────────────
    print("\n─── Phase 1: head training ───")
    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = _run_epoch(model, train_dl, criterion, optimizer, device, training=True)
        val_f1, _, _ = evaluate(model, val_dl, device)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{PHASE1_EPOCHS}  loss={train_loss:.4f}  val_macro-F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Phase 2: unfreeze last block ───────────────────────────────────────────
    print("\n─── Phase 2: fine-tuning last block ───")
    unfreeze_last_block(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR / 5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    no_improve = 0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = _run_epoch(model, train_dl, criterion, optimizer, device, training=True)
        val_f1, _, _ = evaluate(model, val_dl, device)
        scheduler.step()

        improved = val_f1 > best_f1
        marker = " ✓" if improved else ""
        print(f"  Epoch {epoch:02d}/{PHASE2_EPOCHS}  loss={train_loss:.4f}  val_macro-F1={val_f1:.4f}{marker}")

        if improved:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (no improvement for {PATIENCE} epochs)")
                break

    # ── Save best checkpoint ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(
        {'state_dict': best_state, 'num_classes': NUM_CLASSES, 'val_macro_f1': best_f1},
        MODEL_PATH,
    )
    print(f"\nBest val macro-F1: {best_f1:.4f}  →  saved to {MODEL_PATH}")

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\n─── Test set evaluation ───")
    model.load_state_dict(best_state)
    model.to(device)
    from src.evaluate import full_report
    full_report(model, test_dl, device, save_path='models/confusion_matrix.png')

    return model


if __name__ == '__main__':
    train()
