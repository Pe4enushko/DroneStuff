from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from CVstuff.processors.contour_pnp_processor import ContourPnPProcessor
from CVstuff.pnp_processor import OutputMode


def _parse_camera_matrix(values: Iterable[float] | None) -> np.ndarray | None:
    if values is None:
        return None

    matrix_values = list(values)
    if len(matrix_values) != 9:
        raise argparse.ArgumentTypeError("Camera matrix requires 9 values")
    return np.array(matrix_values, dtype=np.float64).reshape(3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect rectangular contours and annotate PnP pose.")
    parser.add_argument("video_source", help="Camera index, video file, image file, stream, or URL.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in OutputMode],
        default=OutputMode.WINDOW.value,
        help="Output mode: show a window or write annotated frames to a file.",
    )
    parser.add_argument("--output-file", help="Output video/image path for --mode file.")
    parser.add_argument("--marker-size", type=float, default=1.0, help="Square marker size in your chosen units.")
    parser.add_argument("--min-contour-area", type=float, default=800.0, help="Ignore contours smaller than this.")
    parser.add_argument(
        "--max-contour-area-ratio",
        type=float,
        default=0.02,
        help="Ignore contours larger than this fraction of the frame area.",
    )
    parser.add_argument("--border-margin", type=int, default=2, help="Reject rectangles this close to the frame edge.")
    parser.add_argument(
        "--approx-epsilon-factor",
        type=float,
        default=0.02,
        help="Contour approximation epsilon as a fraction of contour perimeter.",
    )
    parser.add_argument("--canny-low", type=int, default=50, help="Lower Canny edge threshold.")
    parser.add_argument("--canny-high", type=int, default=150, help="Upper Canny edge threshold.")
    parser.add_argument("--blur-size", type=int, default=5, help="Gaussian blur kernel size; even values are rounded up.")
    parser.add_argument(
        "--morph-kernel-size",
        type=int,
        default=5,
        help="Morphological close kernel size for reconnecting edge gaps; use 1 to disable.",
    )
    parser.add_argument("--max-detections", type=int, default=1, help="Maximum scored rectangles to draw; use 0 for all.")
    parser.add_argument(
        "--use-min-area-fallback",
        dest="use_min_area_fallback",
        action="store_true",
        help="Also fit minAreaRect for plausible contours that do not approximate to exactly four corners.",
    )
    parser.add_argument(
        "--no-min-area-fallback",
        dest="use_min_area_fallback",
        action="store_false",
        help="Only accept contours that approximate to exactly four corners.",
    )
    parser.set_defaults(use_min_area_fallback=True)
    parser.add_argument(
        "--no-edges-window",
        action="store_true",
        help="Do not show the Canny/morphology debug window in window mode.",
    )
    parser.add_argument(
        "--camera-matrix",
        nargs=9,
        type=float,
        metavar=("FX", "S", "CX", "S2", "FY", "CY", "P1", "P2", "P3"),
        help="Optional 3x3 camera matrix values in row-major order.",
    )
    args = parser.parse_args()

    processor = ContourPnPProcessor(
        video_source=args.video_source,
        output_mode=args.mode,
        output_file=args.output_file,
        camera_matrix=_parse_camera_matrix(args.camera_matrix),
        marker_size=args.marker_size,
        min_contour_area=args.min_contour_area,
        max_contour_area_ratio=args.max_contour_area_ratio,
        border_margin=args.border_margin,
        approx_epsilon_factor=args.approx_epsilon_factor,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        blur_size=args.blur_size,
        morph_kernel_size=args.morph_kernel_size,
        max_detections=args.max_detections,
        use_min_area_fallback=args.use_min_area_fallback,
        show_edges_window=not args.no_edges_window,
    )
    processor.run()


if __name__ == "__main__":
    main()
