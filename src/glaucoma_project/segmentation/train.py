"""Train the disc/cup U-Net. Usage: python -m glaucoma_project.segmentation.train [--config PATH]"""
import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from glaucoma_project import REPO_ROOT
from glaucoma_project.data import make_loader
from glaucoma_project.evaluation.metrics import dice
from glaucoma_project.segmentation.losses import DiceBCELoss
from glaucoma_project.segmentation.model import DiscCupUNet

RUNS_DIR = REPO_ROOT / "runs" / "segmentation"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    """Mean Dice for disc and cup over a split, using a 0.5 probability threshold."""
    model.eval()
    scores = []
    for batch in loader:
        predicted = torch.sigmoid(model(batch["image"].to(device))) > 0.5
        scores.append(dice(predicted, batch["mask"].to(device)).cpu())
    scores = torch.cat(scores).mean(dim=0)
    return {"val_dice_disc": scores[0].item(), "val_dice_cup": scores[1].item()}


def train(config: dict, config_path: Path) -> Path:
    set_seed(config["seed"])
    device = pick_device()
    run_dir = RUNS_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                            text=True).stdout.strip()
    (run_dir / "run.json").write_text(json.dumps({"git_commit": commit, "device": str(device)}))

    train_loader = make_loader("train", config["batch_size"], num_workers=config["num_workers"])
    val_loader = make_loader("validation", config["batch_size"], num_workers=config["num_workers"])
    model = DiscCupUNet(config["encoder"], config["pretrained"]).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history, best = [], -1.0
    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            loss = loss_fn(model(batch["image"].to(device)), batch["mask"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        scores = validate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **scores,
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        mean_dice = (scores["val_dice_disc"] + scores["val_dice_cup"]) / 2
        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(row, flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "segmentation.yaml")
    args = parser.parse_args()
    train(yaml.safe_load(args.config.read_text()), args.config)
