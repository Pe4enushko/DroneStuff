from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from pnp_processor import OpenCVPnPProcessor, OutputMode


def _parse_camera_matrix(values: Iterable[float] | None) -> np.ndarray | None:
    if values is None:
        return None

    matrix_values = list(values)
    if len(matrix_values) != 9:
        raise argparse.ArgumentTypeError("Camera matrix requires 9 values")
    return np.array(matrix_values, dtype=np.float64).reshape(3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect polygon contours and annotate PnP pose.")
    parser.add_argument("video_source", help="Camera index, video file, image stream, or URL.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in OutputMode],
        default=OutputMode.WINDOW.value,
        help="Output mode: show a window or write annotated frames to a file.",
    )
    parser.add_argument("--output-file", help="Output video path for --mode file.")
    parser.add_argument("--marker-size", type=float, default=1.0, help="Square marker size in your chosen units.")
    parser.add_argument("--y-margin", type=int, default=20, help="Ignore Harris points this close to top/bottom border.")
    parser.add_argument("--harris-quality", type=float, default=0.01, help="Harris response threshold multiplier.")
    parser.add_argument(
        "--dense-group-radius",
        type=int,
        default=12,
        help="Pixel radius used to group nearby Harris points before fitting the rectangle; use 0 to disable.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold for homography estimation.",
    )
    parser.add_argument(
        "--no-normalized-window",
        action="store_true",
        help="Do not show the normalized grayscale debug window in window mode.",
    )
    parser.add_argument(
        "--camera-matrix",
        nargs=9,
        type=float,
        metavar=("FX", "S", "CX", "S2", "FY", "CY", "P1", "P2", "P3"),
        help="Optional 3x3 camera matrix values in row-major order.",
    )
    args = parser.parse_args()

    processor = OpenCVPnPProcessor(
        video_source=args.video_source,
        output_mode=args.mode,
        output_file=args.output_file,
        camera_matrix=_parse_camera_matrix(args.camera_matrix),
        marker_size=args.marker_size,
        y_margin=args.y_margin,
        harris_quality=args.harris_quality,
        dense_group_radius=args.dense_group_radius,
        ransac_reproj_threshold=args.ransac_threshold,
        show_normalized_window=not args.no_normalized_window,
    )
    processor.run()


if __name__ == "__main__":
    main()
