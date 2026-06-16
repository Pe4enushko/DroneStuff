from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ProcessResult:
    vector: tuple[float, float]
    perspective_angle: float
    image_center: tuple[float, float]
    rectangle_center: tuple[float, float]
    corners: np.ndarray
    debug_path: Path


class RectangleProcessor:
    """Find a rectangle from a PIL image using grayscale adaptive threshold, Harris, and RANSAC."""

    def __init__(
        self,
        debug_dir: str | Path = "debug",
        resize_width: int = 640,
        adaptive_block_size: int = 31,
        adaptive_c: int = 5,
        harris_block_size: int = 2,
        harris_ksize: int = 5,
        harris_k: float = 0.03,
        harris_quality: float = 0.03,
        max_harris_points: int = 180,
        ransac_iterations: int = 1200,
        edge_inlier_distance: float = 5.0,
        min_quad_area_ratio: float = 0.001,
        min_side_ratio: float = 0.25,
        min_corner_angle: float = 25.0,
        border_margin: int = 10,
        random_seed: int = 7,
    ) -> None:
        self.debug_dir = Path(debug_dir)
        self.resize_width = resize_width
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.harris_block_size = harris_block_size
        self.harris_ksize = harris_ksize
        self.harris_k = harris_k
        self.harris_quality = harris_quality
        self.max_harris_points = max_harris_points
        self.ransac_iterations = ransac_iterations
        self.edge_inlier_distance = edge_inlier_distance
        self.min_quad_area_ratio = min_quad_area_ratio
        self.min_side_ratio = min_side_ratio
        self.min_corner_angle = min_corner_angle
        self.border_margin = border_margin
        self.random_seed = random_seed

    def process(self, image: Image.Image, debug_name: str = "detected_rectangle.png") -> ProcessResult:
        original = self._pil_to_bgr(image)
        small, scale = self._resize_for_detection(original)
        threshold = self._adaptive_threshold(small)
        harris_points = self._harris_points(threshold)
        harris_points = self._filter_border_points(harris_points, threshold.shape)

        if len(harris_points) < 4:
            result = self._empty_result(original, debug_name, "Not enough Harris points")
            print(f"x={result.vector[0]:.2f}")
            print(f"y={result.vector[1]:.2f}")
            print(f"perspective_angle={result.perspective_angle:.2f}")
            return result

        small_corners = self._ransac_rectangle(harris_points, threshold.shape)
        if small_corners is None:
            result = self._empty_result(original, debug_name, "Rectangle not found")
            print(f"x={result.vector[0]:.2f}")
            print(f"y={result.vector[1]:.2f}")
            print(f"perspective_angle={result.perspective_angle:.2f}")
            return result

        corners = self._order_quad_points(small_corners / scale)
        rectangle_center = tuple(np.mean(corners, axis=0).astype(float))
        height, width = original.shape[:2]
        image_center = (width / 2.0, height / 2.0)
        vector = (
            rectangle_center[0] - image_center[0],
            rectangle_center[1] - image_center[1],
        )
        perspective_angle = self._diagonal_perspective_angle(corners)

        debug_path = self._save_debug_image(
            original,
            corners,
            image_center,
            rectangle_center,
            vector,
            perspective_angle,
            debug_name,
        )

        result = ProcessResult(
            vector=vector,
            perspective_angle=perspective_angle,
            image_center=image_center,
            rectangle_center=rectangle_center,
            corners=corners,
            debug_path=debug_path,
        )
        print(f"x={result.vector[0]:.2f}")
        print(f"y={result.vector[1]:.2f}")
        print(f"perspective_angle={result.perspective_angle:.2f}")
        return result

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _resize_for_detection(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        if self.resize_width <= 0 or image.shape[1] <= self.resize_width:
            return image, 1.0

        scale = self.resize_width / float(image.shape[1])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        block_size = max(3, int(self.adaptive_block_size))
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.adaptive_c,
        )

    def _harris_points(self, threshold: np.ndarray) -> np.ndarray:
        response = cv2.cornerHarris(
            np.float32(threshold),
            blockSize=self.harris_block_size,
            ksize=self.harris_ksize,
            k=self.harris_k,
        )
        if response.max() <= 0:
            return np.empty((0, 2), dtype=np.float32)

        local_max = response == cv2.dilate(response, None)
        strong = response > self.harris_quality * response.max()
        y_coords, x_coords = np.where(local_max & strong)
        if len(x_coords) == 0:
            return np.empty((0, 2), dtype=np.float32)

        strengths = response[y_coords, x_coords]
        order = np.argsort(strengths)[::-1][: self.max_harris_points]
        return np.column_stack((x_coords[order], y_coords[order])).astype(np.float32)

    def _ransac_rectangle(self, points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
        if len(points) < 4:
            return None

        rng = np.random.default_rng(self.random_seed)
        min_area = image_shape[0] * image_shape[1] * self.min_quad_area_ratio
        best_score = -1.0
        best_corners: np.ndarray | None = None
        sample_count = min(len(points), self.max_harris_points)
        candidates = points[:sample_count]

        for _ in range(self.ransac_iterations):
            sample_indices = rng.choice(sample_count, size=4, replace=False)
            corners = self._order_quad_points(candidates[sample_indices])
            if not cv2.isContourConvex(corners.astype(np.float32)):
                continue

            area = cv2.contourArea(corners.astype(np.float32))
            if area < min_area:
                continue
            if not self._valid_quad_geometry(corners):
                continue
            if self._touches_border(corners, image_shape):
                continue

            score = self._edge_support_score(corners, points)
            if score > best_score:
                best_score = score
                best_corners = corners

        return best_corners

    def _filter_border_points(self, points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        if self.border_margin <= 0 or len(points) == 0:
            return points
        height, width = image_shape
        x_values = points[:, 0]
        y_values = points[:, 1]
        keep = (
            (x_values > self.border_margin)
            & (y_values > self.border_margin)
            & (x_values < width - 1 - self.border_margin)
            & (y_values < height - 1 - self.border_margin)
        )
        return points[keep]

    def _touches_border(self, corners: np.ndarray, image_shape: tuple[int, int]) -> bool:
        if self.border_margin <= 0:
            return False
        height, width = image_shape
        x_values = corners[:, 0]
        y_values = corners[:, 1]
        return bool(
            np.any(x_values <= self.border_margin)
            or np.any(y_values <= self.border_margin)
            or np.any(x_values >= width - 1 - self.border_margin)
            or np.any(y_values >= height - 1 - self.border_margin)
        )

    def _valid_quad_geometry(self, corners: np.ndarray) -> bool:
        side_lengths = np.array(
            [
                np.linalg.norm(corners[(index + 1) % 4] - corners[index])
                for index in range(4)
            ],
            dtype=np.float32,
        )
        longest_side = float(side_lengths.max())
        shortest_side = float(side_lengths.min())
        if longest_side <= 0 or shortest_side / longest_side < self.min_side_ratio:
            return False

        for index in range(4):
            prev_point = corners[(index - 1) % 4]
            point = corners[index]
            next_point = corners[(index + 1) % 4]
            vector_a = prev_point - point
            vector_b = next_point - point
            norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            if norm_product <= 0:
                return False
            cosine = np.clip(float(np.dot(vector_a, vector_b) / norm_product), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cosine)))
            if angle < self.min_corner_angle or angle > 180.0 - self.min_corner_angle:
                return False

        return True

    def _edge_support_score(self, corners: np.ndarray, points: np.ndarray) -> float:
        edge_distances = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            edge_distances.append(self._point_segment_distances(points, start, end))

        min_distances = np.min(np.vstack(edge_distances), axis=0)
        inliers = np.count_nonzero(min_distances <= self.edge_inlier_distance)
        area = cv2.contourArea(corners.astype(np.float32))
        return float(inliers * 1000.0 + np.sqrt(max(area, 0.0)))

    @staticmethod
    def _point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq <= 0:
            return np.full(len(points), np.inf, dtype=np.float32)

        t = np.clip(((points - start) @ segment) / segment_len_sq, 0.0, 1.0)
        projection = start + t[:, None] * segment
        return np.linalg.norm(points - projection, axis=1)

    def _empty_result(self, image: np.ndarray, debug_name: str, message: str) -> ProcessResult:
        height, width = image.shape[:2]
        image_center = (width / 2.0, height / 2.0)
        debug_path = self._save_debug_image(
            image,
            None,
            image_center,
            image_center,
            (0.0, 0.0),
            0.0,
            debug_name,
            message,
        )
        return ProcessResult(
            vector=(0.0, 0.0),
            perspective_angle=0.0,
            image_center=image_center,
            rectangle_center=image_center,
            corners=np.empty((0, 2), dtype=np.float32),
            debug_path=debug_path,
        )

    def _save_debug_image(
        self,
        image: np.ndarray,
        corners: np.ndarray | None,
        image_center: tuple[float, float],
        rectangle_center: tuple[float, float],
        vector: tuple[float, float],
        perspective_angle: float,
        debug_name: str,
        message: str | None = None,
    ) -> Path:
        output = image.copy()
        image_center_px = tuple(np.rint(image_center).astype(np.int32))
        rectangle_center_px = tuple(np.rint(rectangle_center).astype(np.int32))

        if corners is not None and len(corners) == 4:
            cv2.polylines(output, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            for point in corners.astype(np.int32):
                cv2.circle(output, tuple(point), 5, (0, 0, 255), -1)
            cv2.arrowedLine(output, image_center_px, rectangle_center_px, (0, 255, 255), 2, tipLength=0.2)
        else:
            cv2.putText(output, message or "Rectangle not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.circle(output, image_center_px, 5, (255, 0, 0), -1)
        cv2.circle(output, rectangle_center_px, 5, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"vec=({vector[0]:.1f}, {vector[1]:.1f}) angle={perspective_angle:.1f}",
            (10, output.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = self.debug_dir / debug_name
        if not cv2.imwrite(str(debug_path), output):
            raise RuntimeError(f"Could not write debug image: {debug_path}")
        return debug_path

    @staticmethod
    def _diagonal_perspective_angle(corners: np.ndarray) -> float:
        diag_a = float(np.linalg.norm(corners[0] - corners[2]))
        diag_b = float(np.linalg.norm(corners[1] - corners[3]))
        max_diag = max(diag_a, diag_b)
        if max_diag <= 0:
            return 0.0
        return float(np.degrees(np.arctan2(abs(diag_a - diag_b), max_diag)))

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
