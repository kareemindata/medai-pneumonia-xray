import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.optim import AdamW
from tqdm import tqdm

from .data import build_loaders
from .model import build_model


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "acc": float((preds == targets).mean()),
        "auc": float(roc_auc_score(targets, probs)) if len(set(targets)) > 1 else float("nan"),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "latest").unlink(missing_ok=True) if (Path("runs") / "latest").is_symlink() else None

    train_loader, val_loader, test_loader, classes = build_loaders(
        args.data_dir, batch_size=args.batch_size, image_size=args.image_size,
    )
    print(f"classes: {classes}")

    model = build_model(num_classes=len(classes)).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_auc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=running / n)

        val_metrics = evaluate(model, val_loader, args.device)
        history.append({"epoch": epoch, "train_loss": running / n, "val": val_metrics})
        print(f"epoch {epoch}: val={val_metrics}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model_state": model.state_dict(), "classes": classes}, out_dir / "best.pt")

    test_metrics = evaluate(model, test_loader, args.device)
    print(f"test: {test_metrics}")
    (out_dir / "metrics.json").write_text(json.dumps({"history": history, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
