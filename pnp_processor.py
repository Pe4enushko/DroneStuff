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
    contour: np.ndarray
    polygon: np.ndarray
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None


class OpenCVPnPProcessor:
    """Process a video source and annotate polygon detections with PnP pose."""

    def __init__(
        self,
        video_source: Any,
        output_mode: OutputMode | str,
        output_file: str | Path | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        marker_size: float = 1.0,
        min_contour_area: float = 800.0,
        approx_epsilon_factor: float = 0.02,
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
        self.min_contour_area = min_contour_area
        self.approx_epsilon_factor = approx_epsilon_factor
        self.window_name = window_name
        self.show_normalized_window = show_normalized_window
        self.normalized_window_name = normalized_window_name

        if self.output_mode is OutputMode.FILE and self.output_file is None:
            raise ValueError("output_file is required when output_mode is 'file'")

    def run(self) -> None:
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
                        cv2.imshow(self.normalized_window_name, self.normalize_frame(frame))
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

    def process_frame(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        camera_matrix = camera_matrix if camera_matrix is not None else self._camera_matrix_for_frame(frame)
        detections = self.detect(frame, camera_matrix)
        annotated = frame.copy()

        for detection in detections:
            self._draw_detection(annotated, detection, camera_matrix)

        return annotated

    @staticmethod
    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    def detect(self, frame: np.ndarray, camera_matrix: np.ndarray) -> list[Detection]:
        gray = self.normalize_frame(frame)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        blurred = cv2.bilateralFilter(thresh, 9, 75, 75)
        edges = cv2.Canny(blurred, 50, 150)
        

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, self.approx_epsilon_factor * perimeter, True)

            rvec: np.ndarray | None = None
            tvec: np.ndarray | None = None

            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                image_points = self._order_quad_points(polygon.reshape(4, 2))
                ok, rvec, tvec = cv2.solvePnP(
                    self._object_points(),
                    image_points,
                    camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    rvec = None
                    tvec = None

            detections.append(Detection(contour=contour, polygon=polygon, rvec=rvec, tvec=tvec))

        return detections

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        camera_matrix: np.ndarray,
    ) -> None:
        cv2.drawContours(frame, [detection.polygon], -1, (0, 255, 255), 2)

        if detection.rvec is None or detection.tvec is None:
            return

        image_points = self._order_quad_points(detection.polygon.reshape(4, 2)).astype(np.int32)
        cv2.polylines(frame, [image_points], True, (0, 255, 0), 2)

        axis_points = np.float32(
            [
                [0, 0, 0],
                [self.marker_size, 0, 0],
                [0, self.marker_size, 0],
                [0, 0, -self.marker_size],
            ]
        )
        projected_axis, _ = cv2.projectPoints(
            axis_points,
            detection.rvec,
            detection.tvec,
            camera_matrix,
            self.dist_coeffs,
        )
        origin, x_axis, y_axis, z_axis = projected_axis.reshape(-1, 2).astype(int)

        cv2.line(frame, tuple(origin), tuple(x_axis), (0, 0, 255), 3)
        cv2.line(frame, tuple(origin), tuple(y_axis), (0, 255, 0), 3)
        cv2.line(frame, tuple(origin), tuple(z_axis), (255, 0, 0), 3)

    def _open_capture(self, video_source: Any) -> tuple[Any, bool]:
        if hasattr(video_source, "read"):
            return video_source, False

        source = int(video_source) if isinstance(video_source, str) and video_source.isdigit() else video_source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source!r}")
        return capture, True

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

    def _object_points(self) -> np.ndarray:
        half_size = self.marker_size / 2.0
        return np.array(
            [
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
            ],
            dtype=np.float32,
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
