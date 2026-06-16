from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from CVstuff.pnp_processor import OutputMode


@dataclass(frozen=True)
class ContourDetection:
    contour: np.ndarray
    corners: np.ndarray
    score: float
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None


class ContourPnPProcessor:
    """Detect rectangle candidates from contours and estimate pose from their corners."""

    def __init__(
        self,
        video_source: Any,
        output_mode: OutputMode | str,
        output_file: str | Path | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        marker_size: float = 1.0,
        min_contour_area: float = 800.0,
        max_contour_area_ratio: float = 0.02,
        border_margin: int = 2,
        approx_epsilon_factor: float = 0.02,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_size: int = 5,
        morph_kernel_size: int = 5,
        max_detections: int = 1,
        use_min_area_fallback: bool = True,
        window_name: str = "Contour PnP Processor",
        show_edges_window: bool = True,
        edges_window_name: str = "Contour Edges",
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
        self.max_contour_area_ratio = max_contour_area_ratio
        self.border_margin = border_margin
        self.approx_epsilon_factor = approx_epsilon_factor
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_size = blur_size
        self.morph_kernel_size = morph_kernel_size
        self.max_detections = max_detections
        self.use_min_area_fallback = use_min_area_fallback
        self.window_name = window_name
        self.show_edges_window = show_edges_window
        self.edges_window_name = edges_window_name

        if self.output_mode is OutputMode.FILE and self.output_file is None:
            raise ValueError("output_file is required when output_mode is 'file'")

    def run(self) -> None:
        if self._is_image_source(self.video_source):
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
                annotated, edges = self.process_frame(frame, camera_matrix)

                if self.output_mode is OutputMode.WINDOW:
                    cv2.imshow(self.window_name, annotated)
                    if self.show_edges_window:
                        cv2.imshow(self.edges_window_name, edges)
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
                if self.show_edges_window:
                    cv2.destroyWindow(self.edges_window_name)

    def _run_image_source(self, image_path: Path) -> None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image source: {image_path!s}")

        frame = self._resize_frame(frame)
        camera_matrix = self._camera_matrix_for_frame(frame)
        annotated, edges = self.process_frame(frame, camera_matrix)

        if self.output_mode is OutputMode.FILE:
            assert self.output_file is not None
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.output_file), annotated):
                raise RuntimeError(f"Could not write output image: {self.output_file}")
            return

        try:
            cv2.imshow(self.window_name, annotated)
            if self.show_edges_window:
                cv2.imshow(self.edges_window_name, edges)
            cv2.waitKey(0)
        finally:
            cv2.destroyWindow(self.window_name)
            if self.show_edges_window:
                cv2.destroyWindow(self.edges_window_name)

    def process_frame(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        camera_matrix = camera_matrix if camera_matrix is not None else self._camera_matrix_for_frame(frame)
        detections, edges = self.detect(frame, camera_matrix)
        annotated = frame.copy()

        for detection in detections:
            self._draw_detection(annotated, detection, camera_matrix)

        return annotated, edges

    @staticmethod
    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    def detect(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> tuple[list[ContourDetection], np.ndarray]:
        masks = self._candidate_masks(frame)
        debug_mask = np.bitwise_or.reduce(masks)
        detections: list[ContourDetection] = []
        max_contour_area = frame.shape[0] * frame.shape[1] * self.max_contour_area_ratio

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area or area > max_contour_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue

                polygon = cv2.approxPolyDP(contour, self.approx_epsilon_factor * perimeter, True)
                if len(polygon) == 4 and cv2.isContourConvex(polygon):
                    corners = self._order_quad_points(polygon.reshape(4, 2))
                elif self.use_min_area_fallback:
                    corners = self._rectangle_from_contour(contour)
                else:
                    continue

                if self._touches_border(corners, frame.shape):
                    continue

                score = self._score_quad(contour, corners, frame)
                if score <= 0:
                    continue

                rvec: np.ndarray | None = None
                tvec: np.ndarray | None = None
                ok, candidate_rvec, candidate_tvec = cv2.solvePnP(
                    self._object_points(),
                    corners.astype(np.float32),
                    camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if ok:
                    rvec = candidate_rvec
                    tvec = candidate_tvec

                detections.append(
                    ContourDetection(
                        contour=contour,
                        corners=corners,
                        score=score,
                        rvec=rvec,
                        tvec=tvec,
                    )
                )

        detections.sort(key=lambda detection: detection.score, reverse=True)
        detections = self._dedupe_detections(detections)
        if self.max_detections > 0:
            detections = detections[: self.max_detections]
        return detections, debug_mask

    def _candidate_masks(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        blur_size = max(1, self.blur_size)
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size > 1:
            equalized = cv2.GaussianBlur(equalized, (blur_size, blur_size), 0)

        edges = cv2.Canny(equalized, self.canny_low, self.canny_high)
        masks = [self._close_mask(edges)]

        block_size = max(3, self.morph_kernel_size * 8 + 1)
        if block_size % 2 == 0:
            block_size += 1
        adaptive = cv2.adaptiveThreshold(
            equalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -5,
        )
        masks.append(self._close_mask(adaptive))

        top_hat_kernel_size = max(3, self.morph_kernel_size * 7)
        top_hat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (top_hat_kernel_size, top_hat_kernel_size),
        )
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, top_hat_kernel)
        _, top_hat_mask = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(self._close_mask(top_hat_mask))

        local_kernel_size = max(31, self.morph_kernel_size * 14 + 1)
        if local_kernel_size % 2 == 0:
            local_kernel_size += 1
        local_background = cv2.GaussianBlur(gray, (local_kernel_size, local_kernel_size), 0)
        local_bright = cv2.subtract(gray, local_background)
        _, local_otsu_mask = cv2.threshold(local_bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(self._close_mask(local_otsu_mask))

        local_threshold = max(4, int(local_bright.mean() + local_bright.std()))
        _, local_fixed_mask = cv2.threshold(local_bright, local_threshold, 255, cv2.THRESH_BINARY)
        masks.append(self._close_mask(local_fixed_mask))

        return masks

    def _close_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.morph_kernel_size <= 1:
            return mask
        kernel_size = max(1, self.morph_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _dedupe_detections(detections: list[ContourDetection]) -> list[ContourDetection]:
        unique: list[ContourDetection] = []
        for detection in detections:
            current_box = cv2.boundingRect(detection.corners.astype(np.float32))
            if any(
                ContourPnPProcessor._rect_iou(
                    current_box,
                    cv2.boundingRect(existing.corners.astype(np.float32)),
                )
                > 0.6
                for existing in unique
            ):
                continue
            unique.append(detection)
        return unique

    @staticmethod
    def _rect_iou(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = rect_a
        bx, by, bw, bh = rect_b
        x_left = max(ax, bx)
        y_top = max(ay, by)
        x_right = min(ax + aw, bx + bw)
        y_bottom = min(ay + ah, by + bh)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = aw * ah + bw * bh - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _score_quad(self, contour: np.ndarray, corners: np.ndarray, frame: np.ndarray) -> float:
        contour_area = cv2.contourArea(contour)
        quad_area = cv2.contourArea(corners.astype(np.float32))
        if quad_area <= 0:
            return 0.0

        rectangularity = min(contour_area, quad_area) / max(contour_area, quad_area)
        angle_score = self._right_angle_score(corners)
        fill_score = self._inside_outside_fill_score(frame, corners)
        return float(quad_area * rectangularity * angle_score * fill_score)

    @staticmethod
    def _inside_outside_fill_score(frame: np.ndarray, corners: np.ndarray) -> float:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        inner_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(inner_mask, corners.astype(np.int32), 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        outer_mask = cv2.dilate(inner_mask, kernel)
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        if not np.any(inner_mask) or not np.any(ring_mask):
            return 0.0

        inside_pixels = lab[inner_mask > 0].astype(np.float32)
        outside_pixels = lab[ring_mask > 0].astype(np.float32)
        inside_mean = inside_pixels.mean(axis=0)
        outside_mean = outside_pixels.mean(axis=0)

        inside_std = float(inside_pixels.std(axis=0).mean())
        outside_delta = float(np.linalg.norm(inside_mean - outside_mean))
        return max(0.05, min(25.0, (outside_delta * outside_delta) / ((inside_std + 1.0) * 10.0)))

    def _touches_border(self, corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
        if self.border_margin <= 0:
            return False
        height, width = image_shape[:2]
        x_values = corners[:, 0]
        y_values = corners[:, 1]
        return bool(
            np.any(x_values <= self.border_margin)
            or np.any(y_values <= self.border_margin)
            or np.any(x_values >= width - 1 - self.border_margin)
            or np.any(y_values >= height - 1 - self.border_margin)
        )

    def _rectangle_from_contour(self, contour: np.ndarray) -> np.ndarray:
        rectangle = cv2.minAreaRect(contour)
        return self._order_quad_points(cv2.boxPoints(rectangle))

    @staticmethod
    def _right_angle_score(corners: np.ndarray) -> float:
        scores = []
        for index in range(4):
            prev_point = corners[(index - 1) % 4]
            point = corners[index]
            next_point = corners[(index + 1) % 4]
            vector_a = prev_point - point
            vector_b = next_point - point
            norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            if norm_product <= 0:
                return 0.0
            cosine = abs(float(np.dot(vector_a, vector_b) / norm_product))
            scores.append(max(0.0, 1.0 - cosine))
        return float(np.prod(scores))

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: ContourDetection,
        camera_matrix: np.ndarray,
    ) -> None:
        corners = detection.corners.astype(np.int32)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        for index, point in enumerate(corners):
            cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(index + 1),
                tuple(point + np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"score={detection.score:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if detection.rvec is None or detection.tvec is None:
            return

        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            self.dist_coeffs,
            detection.rvec,
            detection.tvec,
            self.marker_size * 0.5,
        )

    def _object_points(self) -> np.ndarray:
        half_size = self.marker_size / 2.0
        return np.array(
            [
                [-half_size, -half_size, 0.0],
                [half_size, -half_size, 0.0],
                [half_size, half_size, 0.0],
                [-half_size, half_size, 0.0],
            ],
            dtype=np.float32,
        )

    def _open_capture(self, video_source: Any) -> tuple[Any, bool]:
        if hasattr(video_source, "read"):
            return video_source, False

        source = int(video_source) if isinstance(video_source, str) and video_source.isdigit() else video_source
        capture = self._video_capture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source!r}")
        return capture, True

    @staticmethod
    def _video_capture(source: Any) -> cv2.VideoCapture:
        if source == 0 and hasattr(cv2, "CAP_DSHOW"):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)

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
    def _is_image_source(video_source: Any) -> bool:
        if not isinstance(video_source, (str, Path)):
            return False
        return Path(video_source).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

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
