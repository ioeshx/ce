import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
	import matplotlib
except Exception:
	matplotlib = None


def normalize01(arr: np.ndarray) -> np.ndarray:
	lo, hi = np.percentile(arr, [2, 98])
	if hi <= lo:
		return np.zeros_like(arr, dtype=np.float32)
	out = (arr - lo) / (hi - lo)
	return np.clip(out, 0.0, 1.0).astype(np.float32)


def smooth_noise(width: int, height: int, seed: int) -> np.ndarray:
	rng = np.random.default_rng(seed)
	small_h = max(8, height // 32)
	small_w = max(8, width // 32)
	noise_small = (rng.random((small_h, small_w)) * 255).astype(np.uint8)

	noise_img = Image.fromarray(noise_small, mode="L")
	noise_img = noise_img.resize((width, height), resample=Image.Resampling.BICUBIC)
	noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.02))
	noise = np.asarray(noise_img, dtype=np.float32) / 255.0
	return noise


def build_fake_activation(
	image_rgb: np.ndarray,
	blur_radius: float,
	seed: int,
	gamma: float,
) -> np.ndarray:
	r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]
	luminance = 0.299 * r + 0.587 * g + 0.114 * b

	lum_img = Image.fromarray((luminance * 255).astype(np.uint8), mode="L")
	lum_blur = lum_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
	luminance_blur = np.asarray(lum_blur, dtype=np.float32) / 255.0

	local_contrast = np.abs(luminance - luminance_blur)
	dark_prior = 1.0 - luminance
	noise = smooth_noise(width=image_rgb.shape[1], height=image_rgb.shape[0], seed=seed)

	activation = 0.45 * dark_prior + 0.35 * local_contrast + 0.20 * noise
	activation = normalize01(activation)
	activation = np.power(activation, gamma, dtype=np.float32)
	return normalize01(activation)


def fallback_jet(activation: np.ndarray) -> np.ndarray:
	x = np.clip(activation, 0.0, 1.0)
	r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
	g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
	b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
	return np.stack([r, g, b], axis=-1).astype(np.float32)


def build_random_activation(width: int, height: int, seed: int) -> np.ndarray:
	rng = np.random.default_rng(seed)
	base = smooth_noise(width=width, height=height, seed=seed)

	# Add a few smooth blob regions so it looks more like a saliency map.
	grid_x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
	grid_y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
	blobs = np.zeros((height, width), dtype=np.float32)
	blob_count = int(rng.integers(3, 8))

	for _ in range(blob_count):
		cx = float(rng.uniform(0.1, 0.9))
		cy = float(rng.uniform(0.1, 0.9))
		sx = float(rng.uniform(0.06, 0.22))
		sy = float(rng.uniform(0.06, 0.22))
		amp = float(rng.uniform(0.5, 1.2))
		blob = amp * np.exp(-(((grid_x - cx) ** 2) / (2 * sx * sx) + ((grid_y - cy) ** 2) / (2 * sy * sy)))
		blobs += blob.astype(np.float32)

	activation = 0.45 * base + 0.55 * normalize01(blobs)
	activation = normalize01(activation)
	return np.power(activation, 0.85, dtype=np.float32)


def colorize_heatmap(activation: np.ndarray, cmap_name: str) -> np.ndarray:
	if matplotlib is not None:
		# Matplotlib >=3.7: prefer non-deprecated colormap lookup API.
		cmap = matplotlib.colormaps.get_cmap(cmap_name)
		heatmap = cmap(activation)[..., :3]
		return np.clip(heatmap, 0.0, 1.0).astype(np.float32)
	return fallback_jet(activation)


def to_uint8(image: np.ndarray) -> np.ndarray:
	return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_outputs(
	input_img: np.ndarray,
	heatmap_img: np.ndarray,
	output_path: Path,
	mode: str,
	alpha: float,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if mode == "heatmap":
		Image.fromarray(to_uint8(heatmap_img)).save(output_path)
		return

	overlay = (1.0 - alpha) * input_img + alpha * heatmap_img

	if mode == "overlay":
		Image.fromarray(to_uint8(overlay)).save(output_path)
		return

	# mode == "both": save two files with suffixes.
	stem = output_path.stem
	suffix = output_path.suffix or ".png"
	parent = output_path.parent
	heatmap_path = parent / f"{stem}_heatmap{suffix}"
	overlay_path = parent / f"{stem}_overlay{suffix}"
	Image.fromarray(to_uint8(heatmap_img)).save(heatmap_path)
	Image.fromarray(to_uint8(overlay)).save(overlay_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate a visually plausible (non-semantic) heatmap for figure illustration."
	)
	parser.add_argument("--input", type=Path, help="Input image path")
	parser.add_argument("--output", type=Path, help="Output path")
	parser.add_argument(
		"--random",
		action="store_true",
		help="Generate a random heatmap without an input image",
	)
	parser.add_argument("--width", type=int, default=512, help="Random heatmap width")
	parser.add_argument("--height", type=int, default=512, help="Random heatmap height")
	parser.add_argument(
		"--mode",
		choices=["heatmap", "overlay", "both"],
		default="overlay",
		help="Output type: pure heatmap, overlay, or both files",
	)
	parser.add_argument(
		"--cmap",
		type=str,
		default="turbo",
		help="Colormap name (uses matplotlib if available; else fallback to jet-like map)",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=0.58,
		help="Overlay strength when mode is overlay/both (0~1)",
	)
	parser.add_argument(
		"--blur",
		type=float,
		default=14.0,
		help="Gaussian blur radius used to create soft regions",
	)
	parser.add_argument(
		"--gamma",
		type=float,
		default=0.9,
		help="Contrast shaping (<1 sharper hot spots, >1 flatter map)",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.output is None:
		raise ValueError("--output is required")
	if not 0.0 <= args.alpha <= 1.0:
		raise ValueError("alpha must be in [0, 1]")
	if args.blur <= 0:
		raise ValueError("blur must be > 0")
	if args.gamma <= 0:
		raise ValueError("gamma must be > 0")
	if args.width <= 0 or args.height <= 0:
		raise ValueError("width and height must be > 0")

	if args.random:
		activation = build_random_activation(width=args.width, height=args.height, seed=args.seed)
		heatmap_np = colorize_heatmap(activation, cmap_name=args.cmap)
		Image.fromarray(to_uint8(heatmap_np)).save(args.output)
		return

	if args.input is None:
		raise ValueError("--input is required unless --random is used")
	if not args.input.exists():
		raise FileNotFoundError(f"Input image not found: {args.input}")

	image = Image.open(args.input).convert("RGB")
	image_np = np.asarray(image, dtype=np.float32) / 255.0

	activation = build_fake_activation(
		image_rgb=image_np,
		blur_radius=args.blur,
		seed=args.seed,
		gamma=args.gamma,
	)
	heatmap_np = colorize_heatmap(activation, cmap_name=args.cmap)

	save_outputs(
		input_img=image_np,
		heatmap_img=heatmap_np,
		output_path=args.output,
		mode=args.mode,
		alpha=args.alpha,
	)


if __name__ == "__main__":
	main()
