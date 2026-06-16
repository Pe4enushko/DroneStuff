from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OutputMode(str, Enum):
    WINDOW = "window"
    FILE = "file"


@dataclass(frozen=True)
class Detection:
    points: np.ndarray
    rectangle: np.ndarray
    homography: np.ndarray
    homography_mask: np.ndarray | None
    rotations: tuple[np.ndarray, ...]
    euler_angles: tuple[tuple[float, float, float], ...]


class OpenCVPnPProcessor:
    """Process a video source and estimate rectangle rotation from Harris corners."""

    def __init__(
        self,
        video_source: Any,
        output_mode: OutputMode | str,
        output_file: str | Path | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        marker_size: float = 1.0,
        y_margin: int = 20,
        harris_quality: float = 0.01,
        ransac_reproj_threshold: float = 5.0,
        dense_group_radius: int = 12,
        window_name: str = "OpenCV PnP Processor",
        show_normalized_window: bool = True,
        normalized_window_name: str = "Normalized Image",
    ) -> None:
        self.video_source = video_source
        self.output_mode = OutputMode(output_mode)
        self.output_file = Path(output_file) if output_file else None
        self.camera_matrix = camera_matrix
        self.dist_coeffs = (
            dist_coeffs.astype(np.float64)
            if dist_coeffs is not None
            else np.zeros((5, 1), dtype=np.float64)
        )
        self.marker_size = marker_size
        self.y_margin = y_margin
        self.harris_quality = harris_quality
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.dense_group_radius = dense_group_radius
        self.window_name = window_name
        self.show_normalized_window = show_normalized_window
        self.normalized_window_name = normalized_window_name

        if self.output_mode is OutputMode.FILE and self.output_file is None:
            raise ValueError("output_file is required when output_mode is 'file'")

    def run(self) -> None:
        if self._is_png_image_source(self.video_source):
            self._run_image_source(Path(self.video_source))
            return

        capture, should_release = self._open_capture(self.video_source)
        writer: cv2.VideoWriter | None = None

        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Could not read the first frame from video_source")

            frame = self._resize_frame(frame)
            frame_size = (frame.shape[1], frame.shape[0])
            camera_matrix = self._camera_matrix_for_frame(frame)

            if self.output_mode is OutputMode.FILE:
                writer = self._open_writer(capture, frame_size)

            while ok and frame is not None:
                annotated = self.process_frame(frame, camera_matrix)

                if self.output_mode is OutputMode.WINDOW:
                    cv2.imshow(self.window_name, annotated)
                    if self.show_normalized_window:
                        cv2.imshow(
                            self.normalized_window_name,
                            self.normalize_frame(cv2.undistort(frame, camera_matrix, self.dist_coeffs)),
                        )
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                else:
                    assert writer is not None
                    writer.write(annotated)

                ok, frame = capture.read()
                if ok and frame is not None:
                    frame = self._resize_frame(frame)
        finally:
            if writer is not None:
                writer.release()
            if should_release:
                capture.release()
            if self.output_mode is OutputMode.WINDOW:
                cv2.destroyWindow(self.window_name)
                if self.show_normalized_window:
                    cv2.destroyWindow(self.normalized_window_name)

    def _run_image_source(self, image_path: Path) -> None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image source: {image_path!s}")

        frame = self._resize_frame(frame)
        camera_matrix = self._camera_matrix_for_frame(frame)
        annotated = self.process_frame(frame, camera_matrix)

        if self.output_mode is OutputMode.FILE:
            assert self.output_file is not None
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.output_file), annotated):
                raise RuntimeError(f"Could not write output image: {self.output_file}")
            return

        try:
            cv2.imshow(self.window_name, annotated)
            if self.show_normalized_window:
                cv2.imshow(
                    self.normalized_window_name,
                    self.normalize_frame(cv2.undistort(frame, camera_matrix, self.dist_coeffs)),
                )
            cv2.waitKey(0)
        finally:
            cv2.destroyWindow(self.window_name)
            if self.show_normalized_window:
                cv2.destroyWindow(self.normalized_window_name)

    def process_frame(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        camera_matrix = camera_matrix if camera_matrix is not None else self._camera_matrix_for_frame(frame)
        detections = self.detect(frame, camera_matrix)
        annotated = frame.copy()

        for detection in detections:
            self._draw_detection(annotated, detection)

        return annotated

    @staticmethod
    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    def detect(self, frame: np.ndarray, camera_matrix: np.ndarray) -> list[Detection]:
        # undistorted = cv2.undistort(frame, camera_matrix, self.dist_coeffs)
        gray = self.normalize_frame(frame)
        harris_response = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
        harris_response = cv2.dilate(harris_response, None)
        points = self._harris_points(harris_response, gray.shape[0])

        if len(points) < 4:
            return []

        points = self._densest_point_group(points, gray.shape)
        if len(points) < 4:
            return []

        rectangle = self._rectangle_from_points(points)
        homography, homography_mask = cv2.findHomography(
            self._rectangle_model_points(),
            rectangle,
            cv2.RANSAC,
            self.ransac_reproj_threshold,
        )
        if homography is None:
            return []

        _, rotations, _, _ = cv2.decomposeHomographyMat(homography, camera_matrix)
        rotation_tuple = tuple(rotations)
        euler_angles = tuple(self._rotation_matrix_to_euler_angles(rotation) for rotation in rotation_tuple)

        return [
            Detection(
                points=points,
                rectangle=rectangle,
                homography=homography,
                homography_mask=homography_mask,
                rotations=rotation_tuple,
                euler_angles=euler_angles,
            )
        ]

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    def _harris_points(self, harris_response: np.ndarray, image_height: int) -> np.ndarray:
        max_response = 0.01 * harris_response.max()
        if max_response <= 0:
            return np.empty((0, 2), dtype=np.float32)

        threshold = self.harris_quality * max_response
        y_coords, x_coords = np.where(harris_response > threshold)

        if len(x_coords) == 0:
            return np.empty((0, 2), dtype=np.float32)

        points = np.column_stack((x_coords, y_coords)).astype(np.float32)
        y_values = points[:, 1]
        keep_mask = (y_values >= self.y_margin) & (y_values <= image_height - self.y_margin)
        return points[keep_mask]

    def _densest_point_group(self, points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        if self.dense_group_radius <= 0 or len(points) < 4:
            return points

        point_pixels = np.rint(points).astype(np.int32)
        width = image_shape[1]
        height = image_shape[0]
        point_pixels[:, 0] = np.clip(point_pixels[:, 0], 0, width - 1)
        point_pixels[:, 1] = np.clip(point_pixels[:, 1], 0, height - 1)

        point_mask = np.zeros((height, width), dtype=np.uint8)
        point_mask[point_pixels[:, 1], point_pixels[:, 0]] = 255

        kernel_size = self.dense_group_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        grouped_mask = cv2.dilate(point_mask, kernel)
        component_count, labels = cv2.connectedComponents(grouped_mask)
        if component_count <= 1:
            return points

        point_labels = labels[point_pixels[:, 1], point_pixels[:, 0]]
        label_counts = np.bincount(point_labels, minlength=component_count)
        label_counts[0] = 0
        densest_label = int(np.argmax(label_counts))

        if label_counts[densest_label] < 4:
            return points

        return points[point_labels == densest_label]

    def _rectangle_from_points(self, points: np.ndarray) -> np.ndarray:
        rectangle = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rectangle)
        return self._order_quad_points(box)

    def _rectangle_model_points(self) -> np.ndarray:
        half_size = self.marker_size / 2.0
        return np.array(
            [
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _rotation_matrix_to_euler_angles(rotation: np.ndarray) -> tuple[float, float, float]:
        sy = np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
        singular = sy < 1e-6

        if singular:
            x_angle = np.arctan2(-rotation[1, 2], rotation[1, 1])
            y_angle = np.arctan2(-rotation[2, 0], sy)
            z_angle = 0.0
        else:
            x_angle = np.arctan2(rotation[2, 1], rotation[2, 2])
            y_angle = np.arctan2(-rotation[2, 0], sy)
            z_angle = np.arctan2(rotation[1, 0], rotation[0, 0])

        angles = np.degrees([x_angle, y_angle, z_angle])
        return float(angles[0]), float(angles[1]), float(angles[2])

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> None:
        for point in detection.points.astype(np.int32):
            cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)

        rectangle = detection.rectangle.astype(np.int32)
        cv2.polylines(frame, [rectangle], True, (0, 255, 0), 2)

        if not detection.euler_angles:
            return

        roll, pitch, yaw = detection.euler_angles[0]
        cv2.putText(
            frame,
            f"rot x={roll:.1f} y={pitch:.1f} z={yaw:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _open_capture(self, video_source: Any) -> tuple[Any, bool]:
        if hasattr(video_source, "read"):
            return video_source, False

        source = int(video_source) if isinstance(video_source, str) and video_source.isdigit() else video_source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source!r}")
        return capture, True

    @staticmethod
    def _is_png_image_source(video_source: Any) -> bool:
        if not isinstance(video_source, (str, Path)):
            return False
        return Path(video_source).suffix.lower() == ".png"

    def _open_writer(
        self,
        capture: cv2.VideoCapture,
        frame_size: tuple[int, int],
    ) -> cv2.VideoWriter:
        assert self.output_file is not None
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_file), fourcc, fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output file for writing: {self.output_file}")
        return writer

    def _camera_matrix_for_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.camera_matrix is not None:
            return self.camera_matrix.astype(np.float64)

        height, width = frame.shape[:2]
        focal_length = float(max(width, height))
        return np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _order_quad_points(points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        point_sum = points.sum(axis=1)
        point_diff = np.diff(points, axis=1).reshape(-1)

        ordered[0] = points[np.argmin(point_sum)]
        ordered[2] = points[np.argmax(point_sum)]
        ordered[1] = points[np.argmin(point_diff)]
        ordered[3] = points[np.argmax(point_diff)]
        return ordered
