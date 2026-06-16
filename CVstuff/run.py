from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

try:
    from process import RectangleProcessor
except ModuleNotFoundError:
    from CVstuff.process import RectangleProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a rectangle in an image and print center vector variables.")
    parser.add_argument("image_path", help="Input image path.")
    parser.add_argument("--debug-dir", default="debug", help="Folder where debug image is saved.")
    parser.add_argument("--debug-name", default="detected_rectangle.png", help="Debug output image filename.")
    parser.add_argument("--resize-width", type=int, default=640, help="Resize width used for detection; use 0 to disable.")
    parser.add_argument("--adaptive-block-size", type=int, default=31, help="Adaptive threshold block size.")
    parser.add_argument("--adaptive-c", type=int, default=5, help="Adaptive threshold C value.")
    parser.add_argument("--harris-quality", type=float, default=0.03, help="Harris corner response quality threshold.")
    parser.add_argument("--ransac-iterations", type=int, default=1200, help="RANSAC sample count.")
    parser.add_argument("--edge-inlier-distance", type=float, default=5.0, help="Harris point distance to rectangle edge.")
    parser.add_argument("--min-side-ratio", type=float, default=0.25, help="Reject quads with very uneven side lengths.")
    parser.add_argument("--min-corner-angle", type=float, default=25.0, help="Reject quads with very sharp corners.")
    parser.add_argument("--border-margin", type=int, default=10, help="Ignore points and quads close to the frame edge.")
    args = parser.parse_args()

    with Image.open(args.image_path) as image:
        processor = RectangleProcessor(
            debug_dir=args.debug_dir,
            resize_width=args.resize_width,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            harris_quality=args.harris_quality,
            ransac_iterations=args.ransac_iterations,
            edge_inlier_distance=args.edge_inlier_distance,
            min_side_ratio=args.min_side_ratio,
            min_corner_angle=args.min_corner_angle,
            border_margin=args.border_margin,
        )
        result = processor.process(image, debug_name=args.debug_name)

    print(f"vector=({result.vector[0]:.2f}, {result.vector[1]:.2f})")
    print(f"rotation={result.rotation:.2f}")
    print(f"perspective_angle={result.perspective_angle:.2f}")
    print(f"debug_path={Path(result.debug_path)}")


if __name__ == "__main__":
    main()
