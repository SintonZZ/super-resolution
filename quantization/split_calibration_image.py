#!/usr/bin/env python3
"""Split one large RGB image into non-overlapping calibration tiles."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageOps


TileBox = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one large image into fixed-size, non-overlapping PNG tiles "
            "for ONNX PTQ calibration. Incomplete right/bottom borders are dropped."
        )
    )
    parser.add_argument("--input-image", type=Path, required=True, help="Source image path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which calibration PNG tiles are written.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=256,
        help="Tile height in pixels. Default: 256.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=256,
        help="Tile width in pixels. Default: 256.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Maximum number of tiles in row-major order; 0 writes all tiles.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Default: input image stem.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output tiles with the same filenames.",
    )
    return parser.parse_args()


def compute_tile_boxes(
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    max_tiles: int = 0,
) -> List[TileBox]:
    values = (image_width, image_height, tile_width, tile_height)
    if any(int(value) <= 0 for value in values):
        raise ValueError("Image and tile dimensions must be positive.")
    if max_tiles < 0:
        raise ValueError("max_tiles must be non-negative.")
    if image_width < tile_width or image_height < tile_height:
        raise ValueError(
            f"Image {image_width}x{image_height} is smaller than one "
            f"{tile_width}x{tile_height} tile."
        )

    boxes = []
    for top in range(0, image_height - tile_height + 1, tile_height):
        for left in range(0, image_width - tile_width + 1, tile_width):
            boxes.append((left, top, left + tile_width, top + tile_height))
            if max_tiles and len(boxes) >= max_tiles:
                return boxes
    return boxes


def output_paths(boxes: Sequence[TileBox], output_dir: Path, prefix: str) -> List[Path]:
    return [
        output_dir / f"{prefix}_y{top:06d}_x{left:06d}.png"
        for left, top, _, _ in boxes
    ]


def split_calibration_image(
    input_image: Path,
    output_dir: Path,
    tile_width: int = 256,
    tile_height: int = 256,
    max_tiles: int = 0,
    prefix: Optional[str] = None,
    force: bool = False,
) -> Tuple[List[Path], Tuple[int, int], Tuple[int, int]]:
    input_image = Path(input_image)
    output_dir = Path(output_dir)
    if not input_image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")

    with Image.open(input_image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image_width, image_height = image.size
        boxes = compute_tile_boxes(
            image_width,
            image_height,
            tile_width,
            tile_height,
            max_tiles=max_tiles,
        )
        prefix = input_image.stem if prefix is None else prefix.strip()
        if not prefix:
            raise ValueError("prefix must not be empty.")
        paths = output_paths(boxes, output_dir, prefix)
        existing = [path for path in paths if path.exists()]
        if existing and not force:
            raise FileExistsError(
                f"Refusing to overwrite {len(existing)} existing tile(s), for example "
                f"{existing[0]}. Pass --force to overwrite them."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        for box, path in zip(boxes, paths):
            image.crop(box).save(path, format="PNG")

    discarded = (image_width % tile_width, image_height % tile_height)
    return paths, (image_width, image_height), discarded


def main() -> None:
    args = parse_args()
    paths, image_size, discarded = split_calibration_image(
        input_image=args.input_image,
        output_dir=args.output_dir,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        max_tiles=args.max_tiles,
        prefix=args.prefix,
        force=args.force,
    )
    print(f"Input: {args.input_image} ({image_size[0]}x{image_size[1]})")
    print(f"Tile: {args.tile_width}x{args.tile_height}, non-overlapping")
    print(f"Written: {len(paths)} PNG tiles -> {args.output_dir}")
    print(
        "Dropped incomplete border: "
        f"right={discarded[0]} px, bottom={discarded[1]} px"
    )


if __name__ == "__main__":
    main()
