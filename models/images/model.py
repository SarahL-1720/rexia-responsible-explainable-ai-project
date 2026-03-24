import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_metrics(
    all_labels: np.ndarray, all_preds: np.ndarray, all_probs: np.ndarray
) -> dict:
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "f1": f1_score(all_labels, all_preds),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "specificity": tn / (tn + fp + 1e-8),
    }


def print_metrics(metrics: dict, loss: float, phase: str, epoch: int) -> None:
    print(
        f"[{phase}] Epoch {epoch:02d} | Loss {loss:.4f} | "
        f"Acc {metrics['accuracy']:.3f} | F1 {metrics['f1']:.3f} | "
        f"AUC {metrics['auc_roc']:.3f} | Prec {metrics['precision']:.3f} | "
        f"Rec {metrics['recall']:.3f} | Spec {metrics['specificity']:.3f}"
    )


# ── One epoch ─────────────────────────────────────────────────────────────────


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase: str,
    epoch: int,
    global_step: int,  # tracks absolute batch count across epochs
) -> tuple[float, dict, int]:
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    # Running counts for step-level metrics without storing all history
    run_tp = run_tn = run_fp = run_fn = 0

    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=f"{phase} epoch {epoch}", leave=False)
        for images, labels in bar:
            images = images.to(device)
            labels_01 = ((labels + 1) / 2).float().to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels_01)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits).cpu().detach().numpy()
            preds = (probs >= 0.5).astype(int)
            labs = labels_01.cpu().numpy().astype(int)

            # Accumulate for epoch-level metrics
            total_loss += loss.item() * len(labels)
            all_labels.extend(labs)
            all_preds.extend(preds)
            all_probs.extend(probs)

            # ── Running confusion matrix for step-level metrics ────────────
            cm = confusion_matrix(labs, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            run_tn += tn
            run_fp += fp
            run_fn += fn
            run_tp += tp

            step_loss = loss.item()
            step_acc = (run_tp + run_tn) / (run_tp + run_tn + run_fp + run_fn)
            step_precision = run_tp / (run_tp + run_fp + 1e-8)
            step_recall = run_tp / (run_tp + run_fn + 1e-8)
            step_f1 = (
                2 * step_precision * step_recall / (step_precision + step_recall + 1e-8)
            )
            step_spec = run_tn / (run_tn + run_fp + 1e-8)

            # ── Log step-level metrics to MLflow ──────────────────────────
            mlflow.log_metrics(
                {
                    f"{phase}/step_loss": step_loss,
                    f"{phase}/step_accuracy": step_acc,
                    f"{phase}/step_precision": step_precision,
                    f"{phase}/step_recall": step_recall,
                    f"{phase}/step_f1": step_f1,
                    f"{phase}/step_spec": step_spec,
                },
                step=global_step,
            )

            global_step += 1
            bar.set_postfix(
                {
                    "loss": f"{step_loss:.4f}",
                    "acc": f"{step_acc:.3f}",
                    "f1": f"{step_f1:.3f}",
                }
            )

    # ── Epoch-level metrics ────────────────────────────────────────────────
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    print_metrics(metrics, avg_loss, phase.capitalize(), epoch)

    mlflow.log_metrics(
        {
            f"{phase}/epoch_loss": avg_loss,
            f"{phase}/epoch_accuracy": metrics["accuracy"],
            f"{phase}/epoch_f1": metrics["f1"],
            f"{phase}/epoch_auc_roc": metrics["auc_roc"],
            f"{phase}/epoch_precision": metrics["precision"],
            f"{phase}/epoch_recall": metrics["recall"],
            f"{phase}/epoch_specificity": metrics["specificity"],
        },
        step=epoch,
    )

    return avg_loss, metrics, global_step


# ── Training loop ─────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    freeze_backbone: bool = True,
    run_name: str = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
    model.to(device)

    pos_weight = torch.tensor([0.3], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    with mlflow.start_run(run_name=run_name):

        # ── Log hyperparameters ───────────────────────────────────────────
        mlflow.log_params(
            {
                "n_epochs": n_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "freeze_backbone": freeze_backbone,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "model": "MobileNetV3-Small",
            }
        )

        history = {"train": [], "val": []}
        best_val_auc = -1.0
        best_weights = None
        epochs_no_imp = 0
        global_step = 0  # shared counter across all train batches

        for epoch in range(1, n_epochs + 1):
            train_loss, train_metrics, global_step = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                "train",
                epoch,
                global_step,
            )
            val_loss, val_metrics, _ = run_epoch(
                model, val_loader, criterion, None, device, "val", epoch, global_step=0
            )
            scheduler.step()

            # Log current learning rate
            mlflow.log_metric("train/lr", scheduler.get_last_lr()[0], step=epoch)

            history["train"].append({"loss": train_loss, **train_metrics})
            history["val"].append({"loss": val_loss, **val_metrics})

            # ── Early stopping ────────────────────────────────────────────
            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_imp = 0
                mlflow.log_metric("val/best_auc", best_val_auc, step=epoch)
                print(f"  ✓ New best val AUC: {best_val_auc:.4f} — weights saved.")
            else:
                epochs_no_imp += 1
                print(f"  No improvement for {epochs_no_imp}/{patience} epochs.")
                if epochs_no_imp >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    mlflow.set_tag("early_stopped", f"epoch_{epoch}")
                    break

            print()

        # ── Test evaluation ───────────────────────────────────────────────
        model.load_state_dict(best_weights)
        print("=" * 70)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 70)
        test_loss, test_metrics, _ = run_epoch(
            model, test_loader, criterion, None, device, "test", epoch=0, global_step=0
        )

        mlflow.log_metrics(
            {
                "test/loss": test_loss,
                "test/accuracy": test_metrics["accuracy"],
                "test/f1": test_metrics["f1"],
                "test/auc_roc": test_metrics["auc_roc"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/specificity": test_metrics["specificity"],
            }
        )

        # Log the best model weights
        mlflow.pytorch.log_model(model, artifact_path="model")

    return {"history": history, "test_loss": test_loss, "test_metrics": test_metrics}
