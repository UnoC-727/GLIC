
import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import transforms

from Meter import AverageMeterTEST, AverageMeterTRAIN
from glic_model import GLICModel


def setup_logger(log_file: Path) -> None:
    """Initialize a clean root logger that writes to both file and stdout."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logging.info("Logging to %s", log_file)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_2tuple(value) -> Tuple[int, int]:
    """Convert an argparse value into a (H, W) tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected 2 values, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def compute_psnr_from_mse(mse_value: torch.Tensor) -> torch.Tensor:
    """Compute PSNR from MSE assuming inputs are normalized to [0, 1]."""
    eps = 1e-12
    return 10.0 * torch.log10(1.0 / torch.clamp(mse_value, min=eps))


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss for learned image compression."""

    def __init__(self, lmbda: float = 1e-2, loss_type: str = "mse") -> None:
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.mse = nn.MSELoss()

    def forward(self, output: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        n, _, h, w = target.size()
        num_pixels = n * h * w

        bpp_loss = sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values()
        )

        result: Dict[str, torch.Tensor] = {
            "bpp_loss": bpp_loss,
        }

        if self.loss_type == "mse":
            mse_loss = self.mse(output["x_hat"], target)
            result["distortion_loss"] = mse_loss
            result["loss"] = self.lmbda * (255 ** 2) * mse_loss + bpp_loss
            result["psnr"] = compute_psnr_from_mse(mse_loss)
            result["ms_ssim"] = ms_ssim(
                torch.round(output["x_hat"] * 255.0),
                torch.round(target * 255.0),
                data_range=255.0,
                size_average=True,
            )
        else:
            ms_ssim_value = ms_ssim(output["x_hat"], target, data_range=1.0, size_average=True)
            result["distortion_loss"] = 1.0 - ms_ssim_value
            result["loss"] = self.lmbda * result["distortion_loss"] + bpp_loss
            result["psnr"] = torch.tensor(float("nan"), device=target.device)
            result["ms_ssim"] = ms_ssim_value

        return result


class CustomDataParallel(nn.DataParallel):
    """Expose wrapped module attributes directly."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, whether wrapped or not."""
    return model.module if isinstance(model, nn.DataParallel) else model


def configure_optimizers(net: nn.Module, args) -> Tuple[optim.Optimizer, optim.Optimizer]:
    """Build main optimizer and auxiliary optimizer."""
    params_dict = dict(net.named_parameters())

    main_param_names = {
        name
        for name, param in net.named_parameters()
        if param.requires_grad and not name.endswith(".quantiles")
    }
    aux_param_names = {
        name
        for name, param in net.named_parameters()
        if param.requires_grad and name.endswith(".quantiles")
    }

    overlap = main_param_names & aux_param_names
    if overlap:
        raise ValueError(f"Parameters assigned to both optimizers: {sorted(overlap)}")

    optimizer = optim.Adam(
        (params_dict[name] for name in sorted(main_param_names)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[name] for name in sorted(aux_param_names)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def build_transforms(patch_size: Optional[Tuple[int, int]], is_train: bool):
    if is_train:
        if patch_size is None:
            raise ValueError("patch_size must be provided for training transforms.")
        return transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def build_dataloader(
    root: str,
    split: str,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    dataset = ImageFolder(root, split=split, transform=transform)
    use_persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=(device == "cuda"),
        persistent_workers=use_persistent_workers,
    )


def evaluate(epoch_or_step: int, dataloader: DataLoader, model: nn.Module, criterion: RateDistortionLoss) -> float:
    """Run evaluation on the validation set."""
    model.eval()
    device = next(model.parameters()).device

    loss_meter = AverageMeterTEST()
    bpp_meter = AverageMeterTEST()
    distortion_meter = AverageMeterTEST()
    aux_meter = AverageMeterTEST()
    psnr_meter = AverageMeterTEST()
    msssim_meter = AverageMeterTEST()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)

            output = model(batch)
            metrics = criterion(output, batch)
            aux_value = unwrap_model(model).aux_loss()

            loss_meter.update(metrics["loss"])
            bpp_meter.update(metrics["bpp_loss"])
            distortion_meter.update(metrics["distortion_loss"])
            aux_meter.update(aux_value)
            msssim_meter.update(metrics["ms_ssim"])

            if not torch.isnan(metrics["psnr"]):
                psnr_meter.update(metrics["psnr"])

    logging.info(
        "Eval @ %s | Loss: %.4f | Distortion: %.6f | Bpp: %.6f | Aux: %.6f | PSNR: %.4f | MS-SSIM: %.6f",
        epoch_or_step,
        float(loss_meter.avg),
        float(distortion_meter.avg),
        float(bpp_meter.avg),
        float(aux_meter.avg),
        float(psnr_meter.avg) if psnr_meter.count > 0 else float("nan"),
        float(msssim_meter.avg),
    )
    return float(loss_meter.avg)


def save_checkpoint(
    save_dir: Path,
    state: Dict,
    is_best: bool,
    step: Optional[int] = None,
) -> None:
    """Save latest checkpoint, optional periodic checkpoint, and best checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)

    latest_path = save_dir / "checkpoint_latest.pth.tar"
    torch.save(state, latest_path)

    if step is not None:
        step_path = save_dir / f"checkpoint_step_{step}.pth.tar"
        torch.save(state, step_path)

    if is_best:
        best_path = save_dir / "checkpoint_best.pth.tar"
        torch.save(state, best_path)


def load_pretrained(model: nn.Module, pretrained_path: str) -> None:
    logging.info("Loading pretrained weights from %s", pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    checkpoint_path: str,
) -> Tuple[int, int]:
    logging.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    return start_epoch, global_step


def maybe_update_lr(
    epoch: int,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    lr_drop_epoch: int,
    lr_after_drop: float,
    already_dropped: bool,
) -> bool:
    """Apply one-off LR drop when reaching the configured epoch."""
    if already_dropped or epoch < lr_drop_epoch:
        return already_dropped

    for group in optimizer.param_groups:
        group["lr"] = lr_after_drop
    for group in aux_optimizer.param_groups:
        group["lr"] = lr_after_drop

    logging.info("Learning rate dropped to %.6g at epoch %d", lr_after_drop, epoch)
    return True


def count_parameters_in_millions(model: nn.Module) -> float:
    return sum(np.prod(p.size()) for p in model.parameters()) / 1e6


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training script for GLICModel.")

    # Data
    parser.add_argument("--train-root", type=str, default="/home/datasets")
    parser.add_argument("--train-split", type=str, default="flickr")
    parser.add_argument("--test-root", type=str, default="/home/datasets")
    parser.add_argument("--test-split", type=str, default="kodak")

    # Training
    parser.add_argument("-e", "--epochs", default=600, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--lmbda", type=float, default=0.05)
    parser.add_argument("--ortho-weight", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("-n", "--num-workers", type=int, default=16)
    parser.add_argument("--clip-max-norm", default=1.0, type=float)
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "ms-ssim"])

    # Crop / stage schedule
    parser.add_argument("--patch-size", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--large-patch-size", nargs=2, type=int, default=(512, 512))
    parser.add_argument("--large-patch-start-epoch", type=int, default=510)
    parser.add_argument("--lr-drop-epoch", type=int, default=500)
    parser.add_argument("--lr-after-drop", type=float, default=1e-5)

    # Runtime
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Checkpointing
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--save-dir", type=str, default="./ckpt")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metric-update-every", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None)

    # Optional pretrained init
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="/home/checkpoints/0.05checkpoint_best.pth.tar",
    )

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        seed_everything(args.seed)

    torch.backends.cudnn.benchmark = True

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    save_dir = Path(args.save_dir) / f"lambda_{args.lmbda}"
    log_file = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_file)

    logging.info("Arguments:")
    for key, value in sorted(vars(args).items()):
        logging.info("  %s: %s", key, value)

    train_patch_size = to_2tuple(args.patch_size)
    large_patch_size = to_2tuple(args.large_patch_size)
    large_batch_size = max(1, args.batch_size // 4)

    train_transform = build_transforms(train_patch_size, is_train=True)
    test_transform = build_transforms(None, is_train=False)

    train_loader = build_dataloader(
        root=args.train_root,
        split=args.train_split,
        transform=train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    test_loader = build_dataloader(
        root=args.test_root,
        split=args.test_split,
        transform=test_transform,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )

    model = GLICModel().to(device)
    optimizer, aux_optimizer = configure_optimizers(model, args)
    criterion = RateDistortionLoss(lmbda=args.lmbda, loss_type=args.loss_type)

    start_epoch = 0
    global_step = 0

    if args.pretrained:
        load_pretrained(model, args.pretrained_path)

    if args.checkpoint:
        start_epoch, global_step = load_checkpoint(model, optimizer, aux_optimizer, args.checkpoint)

    if args.cuda and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)

    if args.test_only:
        evaluate(0, test_loader, model, criterion)
        return

    logging.info("Number of parameters: %.3f M", count_parameters_in_millions(model))

    elapsed_meter, data_time_meter, loss_meter, psnr_meter, bpp_meter, distortion_meter, ortho_meter = [
        AverageMeterTRAIN(200) for _ in range(7)
    ]

    best_loss = evaluate(0, test_loader, model, criterion)
    lr_dropped = start_epoch >= args.lr_drop_epoch
    large_patch_enabled = start_epoch >= args.large_patch_start_epoch

    if lr_dropped:
        for group in optimizer.param_groups:
            group["lr"] = args.lr_after_drop
        for group in aux_optimizer.param_groups:
            group["lr"] = args.lr_after_drop

    if large_patch_enabled:
        train_transform = build_transforms(large_patch_size, is_train=True)
        train_loader = build_dataloader(
            root=args.train_root,
            split=args.train_split,
            transform=train_transform,
            batch_size=large_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            device=device,
        )
        logging.info(
            "Resume directly in large-patch stage: patch_size=%s, batch_size=%d",
            large_patch_size,
            large_batch_size,
        )

    for epoch in range(start_epoch, args.epochs):
        lr_dropped = maybe_update_lr(
            epoch=epoch,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            lr_drop_epoch=args.lr_drop_epoch,
            lr_after_drop=args.lr_after_drop,
            already_dropped=lr_dropped,
        )

        if (not large_patch_enabled) and epoch >= args.large_patch_start_epoch:
            train_transform = build_transforms(large_patch_size, is_train=True)
            train_loader = build_dataloader(
                root=args.train_root,
                split=args.train_split,
                transform=train_transform,
                batch_size=large_batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                device=device,
            )
            large_patch_enabled = True
            logging.info(
                "Switched to large-patch stage at epoch %d: patch_size=%s, batch_size=%d",
                epoch,
                large_patch_size,
                large_batch_size,
            )

        model.train()
        data_end_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            data_time_meter.update(time.time() - data_end_time)
            step_start_time = time.time()

            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)

            output = model(batch)
            metrics = criterion(output, batch)

            ortho_loss = unwrap_model(model).ortho_loss()
            total_loss = metrics["loss"] + args.ortho_weight * ortho_loss
            total_loss.backward()

            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

            optimizer.step()

            aux_loss = unwrap_model(model).aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if batch_idx % args.metric_update_every == 0:
                elapsed_meter.update(time.time() - step_start_time)
                loss_meter.update(float(metrics["loss"].item()))
                bpp_meter.update(float(metrics["bpp_loss"].item()))
                distortion_meter.update(float(metrics["distortion_loss"].item()))
                ortho_meter.update(float(ortho_loss.item()))

                if not torch.isnan(metrics["psnr"]):
                    psnr_meter.update(float(metrics["psnr"].item()))

            if global_step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    "Epoch %d | Step %d | %d/%d samples | "
                    "Time %.3f (%.3f) | Data %.3f (%.3f) | "
                    "Loss %.4f (%.4f) | Ortho %.4f (%.4f) | "
                    "PSNR %.4f (%.4f) | Bpp %.6f (%.6f) | Distortion %.6f (%.6f) | LR %.6g",
                    epoch,
                    global_step,
                    batch_idx * len(batch),
                    len(train_loader.dataset),
                    elapsed_meter.val,
                    elapsed_meter.avg,
                    data_time_meter.val,
                    data_time_meter.avg,
                    loss_meter.val,
                    loss_meter.avg,
                    ortho_meter.val,
                    ortho_meter.avg,
                    psnr_meter.val,
                    psnr_meter.avg,
                    bpp_meter.val,
                    bpp_meter.avg,
                    distortion_meter.val,
                    distortion_meter.avg,
                    current_lr,
                )

            if global_step > 0 and global_step % args.eval_every == 0:
                eval_loss = evaluate(global_step, test_loader, model, criterion)
                is_best = eval_loss < best_loss
                best_loss = min(best_loss, eval_loss)

                logging.info("Best eval loss: %.6f | Current eval loss: %.6f", best_loss, eval_loss)

                if args.save:
                    save_checkpoint(
                        save_dir=save_dir,
                        state={
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": unwrap_model(model).state_dict(),
                            "loss": eval_loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                        },
                        is_best=is_best,
                        step=global_step,
                    )

            global_step += 1
            data_end_time = time.time()

    logging.info("Training finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
