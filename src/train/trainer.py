import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.data.cifar import get_cifar10, get_cifar10_notransform
from src.data.dataset_wrapper import IndexedDataset, make_subset_loader
from src.models.resnet import get_resnet18
from src.scoring.loss_score import LossScorer
from src.pruning.random_pruner import RandomPruner
from src.pruning.raw_topk_pruner import RawTopKPruner
from src.pruning.calibrated_topk_pruner import CalibratedTopKPruner
from src.pruning.metrics import score_drift_index, mean_turnover
from src.eval.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.io import (
    ensure_dir, save_config, save_json, save_scores, save_masks, MetricsLogger
)
from src.utils.logging import setup_logger
from src.utils.config import generate_run_name


def _build_pruner(config: dict):
    mode = config["pruning"]["mode"]
    if mode == "full":
        return None
    elif mode == "random_pruning":
        return RandomPruner()
    elif mode == "raw_topk_loss":
        return RawTopKPruner()
    elif mode == "calibrated_topk_loss":
        return CalibratedTopKPruner(
            window_size=config["pruning"].get("local_window", 2),
            ema_alpha=config["pruning"].get("ema_alpha", 0.8),
        )
    else:
        raise ValueError(f"Unknown pruning mode: {mode}")


def _build_scheduler(optimizer, config: dict):
    scheduler_type = config.get("scheduler", "multistep")
    epochs = config["epochs"]
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        milestones = config.get("milestones", [int(epochs * 0.5), int(epochs * 0.75)])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


def _is_pruning_epoch(epoch: int, config: dict) -> bool:
    p = config["pruning"]
    if p["mode"] == "full":
        return False
    if not p.get("enabled", True):
        return False
    warmup = p.get("warmup_epochs", 10)
    interval = p.get("interval_epochs", 5)
    if epoch < warmup:
        return False
    return (epoch - warmup) % interval == 0


def train(config: dict) -> dict:
    seed = config.get("seed", 0)
    set_seed(seed)

    run_name = generate_run_name(config)
    output_dir = Path(config.get("output_dir", "outputs")) / run_name
    ensure_dir(output_dir)

    logger = setup_logger("sbdp", log_file=output_dir / "train.log")
    logger.info(f"Run: {run_name}")
    logger.info(f"Config: {config}")

    save_config(config, output_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    train_dataset, test_dataset = get_cifar10(config.get("data_dir", "./data"))
    train_indexed = IndexedDataset(train_dataset)
    test_indexed = IndexedDataset(test_dataset)

    # Scoring dataset (no augmentation for consistent scores)
    score_dataset_raw = get_cifar10_notransform(config.get("data_dir", "./data"))
    score_indexed = IndexedDataset(score_dataset_raw)

    total_samples = len(train_indexed)
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 2)

    test_loader = make_subset_loader(
        test_indexed, None, batch_size, shuffle=False, num_workers=num_workers
    )

    # Model
    num_classes = config.get("num_classes", 10)
    model = get_resnet18(num_classes).to(device)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 5e-4),
    )
    scheduler = _build_scheduler(optimizer, config)

    # Pruner
    pruner = _build_pruner(config)
    scorer = LossScorer()
    pruner_state = {}

    # State
    selected_ids = None  # None = use all
    score_history = []
    mask_history = []
    mode = config["pruning"]["mode"]
    retention_ratio = config["pruning"].get("retention_ratio", 1.0)

    metrics_logger = MetricsLogger(output_dir / "metrics.csv")
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    epochs = config.get("epochs", 100)

    for epoch in range(epochs):
        # Build train loader for current subset
        train_loader = make_subset_loader(
            train_indexed, selected_ids, batch_size, shuffle=True, num_workers=num_workers
        )
        current_subset_size = len(selected_ids) if selected_ids is not None else total_samples

        # Train one epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, sample_ids in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate
        test_result = evaluate(model, test_loader, device)
        test_loss = test_result["loss"]
        test_acc = test_result["acc"]

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        scheduler.step()

        # Log
        metrics_logger.log({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "current_subset_size": current_subset_size,
            "mode": mode,
            "seed": seed,
        })

        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"subset={current_subset_size}/{total_samples}"
        )

        # Pruning step
        if _is_pruning_epoch(epoch, config):
            logger.info(f"Pruning at epoch {epoch} (mode={mode}, retention={retention_ratio})")

            # Score ALL samples (using no-augmentation dataset)
            score_loader = make_subset_loader(
                score_indexed, None, batch_size, shuffle=False, num_workers=num_workers
            )
            scores = scorer.compute_scores(model, score_loader, device)

            # Record score history
            score_history.append({"epoch": epoch, "scores": dict(scores)})

            # Select subset
            selected_ids = pruner.select(scores, retention_ratio, state=pruner_state)

            # Record mask history
            mask_history.append({"epoch": epoch, "selected_ids": list(selected_ids)})

            logger.info(f"Selected {len(selected_ids)}/{total_samples} samples")

    # Save artifacts
    metrics_logger.save()
    save_scores(score_history, output_dir / "score_history.pt")
    save_masks(mask_history, output_dir / "mask_history.pt")
    torch.save(model.state_dict(), output_dir / "final_checkpoint.pt")

    # Compute stability metrics
    sdi = score_drift_index(score_history) if len(score_history) >= 2 else 0.0
    mt = mean_turnover(mask_history) if len(mask_history) >= 2 else 0.0

    summary = {
        "run_name": run_name,
        "mode": mode,
        "seed": seed,
        "retention_ratio": retention_ratio,
        "epochs": epochs,
        "best_test_acc": round(best_test_acc, 6),
        "final_test_acc": round(test_acc, 6),
        "mean_score_drift": round(sdi, 6),
        "mean_turnover": round(mt, 6),
        "num_pruning_events": len(mask_history),
        "total_samples": total_samples,
    }
    save_json(summary, output_dir / "summary.json")
    logger.info(f"Summary: {summary}")

    return summary
