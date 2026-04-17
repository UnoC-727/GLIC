import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from glic_model import GLICModel


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more GLICModel checkpoints with forward-estimated BPP and actual compressed BPP."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint paths.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size for forward pass. Compress/decompress is always run with batch size 1 for clarity and robustness.",
    )
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=64,
        help="Pad image height/width to a multiple of this value before encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images under --image-dir.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary as JSON.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CuDNN behavior.",
    )
    return parser.parse_args()


def setup_runtime(deterministic: bool) -> None:
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, recursive: bool = False):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

        glob_fn = self.image_dir.rglob if recursive else self.image_dir.glob
        self.paths = sorted(
            p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found in: {image_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.to_tensor(image)
        return image, str(path)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n


@dataclass
class EvalResult:
    bpp: float
    psnr: float


@dataclass
class CheckpointResult:
    checkpoint: str
    forward: EvalResult
    compress: EvalResult


@dataclass
class SummaryResult:
    checkpoints: List[CheckpointResult]


def get_padding(height: int, width: int, multiple: int) -> Tuple[int, int, int, int]:
    padded_h = math.ceil(height / multiple) * multiple
    padded_w = math.ceil(width / multiple) * multiple
    pad_left = 0
    pad_right = padded_w - width
    pad_top = 0
    pad_bottom = padded_h - height
    return pad_left, pad_right, pad_top, pad_bottom



def pad_image(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, height, width = x.shape
    padding = get_padding(height, width, multiple)
    pad_left, pad_right, pad_top, pad_bottom = padding
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return x_padded, padding



def crop_image(x: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = padding
    h_end = x.shape[-2] - pad_bottom if pad_bottom > 0 else x.shape[-2]
    w_end = x.shape[-1] - pad_right if pad_right > 0 else x.shape[-1]
    return x[..., pad_top:h_end, pad_left:w_end]



def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)



def estimate_bpp_from_likelihoods(
    likelihoods_dict: Dict[str, torch.Tensor], num_pixels: int
) -> float:
    total_bits = 0.0
    for likelihoods in likelihoods_dict.values():
        total_bits += torch.log(likelihoods).sum().item() / (-math.log(2))
    return total_bits / num_pixels



def count_bytes_in_strings(strings: Sequence) -> int:
    total_bytes = 0
    for group in strings:
        if isinstance(group, (list, tuple)):
            for item in group:
                total_bytes += len(item)
        else:
            total_bytes += len(group)
    return total_bytes



def load_state_dict_from_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid state_dict in checkpoint: {checkpoint_path}")

    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    return state_dict



def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def evaluate_forward(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_multiple: int,
) -> EvalResult:
    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()

    for images, _paths in tqdm(dataloader, desc="Forward", leave=False):
        images = images.to(device, non_blocking=True)

        padded_images, padding = pad_image(images, multiple=pad_multiple)
        outputs = model(padded_images)
        recon = crop_image(outputs["x_hat"], padding).clamp_(0.0, 1.0)

        _, _, height, width = images.shape
        num_pixels = height * width

        cur_psnr = compute_psnr(images, recon)
        cur_bpp = estimate_bpp_from_likelihoods(outputs["likelihoods"], num_pixels)

        batch_size = images.size(0)
        avg_psnr.update(cur_psnr, batch_size)
        avg_bpp.update(cur_bpp, batch_size)

    return EvalResult(bpp=avg_bpp.avg, psnr=avg_psnr.avg)


@torch.no_grad()
def evaluate_compress(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    pad_multiple: int,
    num_workers: int,
) -> EvalResult:
    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()

    single_loader = build_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for images, _paths in tqdm(single_loader, desc="Compress", leave=False):
        images = images.to(device, non_blocking=True)

        padded_images, padding = pad_image(images, multiple=pad_multiple)
        encoded = model.compress(padded_images)
        decoded = model.decompress(encoded["strings"], encoded["shape"])
        recon = crop_image(decoded["x_hat"], padding).clamp_(0.0, 1.0)

        _, _, height, width = images.shape
        num_pixels = height * width

        cur_psnr = compute_psnr(images, recon)
        total_bytes = count_bytes_in_strings(encoded["strings"])
        cur_bpp = total_bytes * 8.0 / num_pixels

        avg_psnr.update(cur_psnr)
        avg_bpp.update(cur_bpp)

    return EvalResult(bpp=avg_bpp.avg, psnr=avg_psnr.avg)



def evaluate_checkpoint(
    checkpoint_path: str,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> CheckpointResult:
    logging.info("Loading checkpoint: %s", checkpoint_path)

    model = GLICModel().to(device)
    state_dict = load_state_dict_from_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.update()
    model.eval()

    forward_loader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    forward_result = evaluate_forward(
        model=model,
        dataloader=forward_loader,
        device=device,
        pad_multiple=args.pad_multiple,
    )
    compress_result = evaluate_compress(
        model=model,
        dataset=dataset,
        device=device,
        pad_multiple=args.pad_multiple,
        num_workers=args.num_workers,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return CheckpointResult(
        checkpoint=checkpoint_path,
        forward=forward_result,
        compress=compress_result,
    )



def print_result(result: CheckpointResult) -> None:
    print("=" * 60)
    print(f"Checkpoint: {result.checkpoint}")
    print(f"Forward  : BPP = {result.forward.bpp:.6f}, PSNR = {result.forward.psnr:.4f}")
    print(f"Compress : BPP = {result.compress.bpp:.6f}, PSNR = {result.compress.psnr:.4f}")



def save_summary_json(summary: SummaryResult, save_path: str) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    logging.info("Saved JSON summary to %s", output_path)



def main() -> None:
    args = parse_args()
    setup_runtime(args.deterministic)
    setup_logger()

    device = torch.device(args.device)
    logging.info("Arguments: %s", vars(args))
    logging.info("Using device: %s", device)

    dataset = ImageDataset(args.image_dir, recursive=args.recursive)
    logging.info("Found %d images in %s", len(dataset), args.image_dir)

    checkpoint_results: List[CheckpointResult] = []
    for checkpoint_path in tqdm(args.checkpoint, desc="Checkpoints"):
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            args=args,
            device=device,
        )
        checkpoint_results.append(result)
        print_result(result)

    print("\n" + "=" * 60)
    print("Final Results")
    for result in checkpoint_results:
        print_result(result)

    if args.save_json is not None:
        save_summary_json(SummaryResult(checkpoints=checkpoint_results), args.save_json)


if __name__ == "__main__":
    main()
