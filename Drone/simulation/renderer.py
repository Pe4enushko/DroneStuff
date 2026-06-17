from .config import Config
from .utils import clamp
from .detection import RectangleProcessor
from pathlib import Path
import moderngl as mgl
import numpy as np
import pygame as pg
from PIL import Image, ImageChops


class Renderer:
    def __init__(self, field, drone, camera, drone_camera):
        self.config = Config()

        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)

        pg.display.set_mode((self.config.WINDOW_W, self.config.WINDOW_H), pg.OPENGL | pg.DOUBLEBUF)

        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.viewport = (0, 0, self.config.WINDOW_W, self.config.WINDOW_H)

        self.field = field
        self.drone = drone
        self.camera = camera
        self.drone_camera = drone_camera
        self.rect_processor = RectangleProcessor()
        self._update_contexts()

        self.clock = pg.time.Clock()
        self.fps = self.config.FPS

        self.running = True
        self.paused = False

        self.move_speed = Config.SPEED
        self.mouse_sensitivity = Config.MOUSE_SENSITIVITY
        self.zoom_speed = Config.ZOOM_SPEED

        pg.mouse.set_visible(False)
        pg.event.set_grab(True)

        # ---- Параметры плавной посадки ----
        self.landing_target_pos = None
        self.landing_target_rot = None
        self.landing_progress = 0.0
        self.landing_duration = 240         # секунды
        self.is_landing = False

    def _update_contexts(self):
        self.field.ctx = self.ctx
        self.field.create_buffers()

        self.drone.ctx = self.ctx
        self.drone.create_buffers()

        self.camera.ctx = self.ctx
        self.camera.create_shaders()

        self.drone_camera.ctx = self.ctx
        self.drone_camera.create_shaders()

    def _get_scan_data(self):
        img = self.drone_camera.get_scan()
        height_map, max_dist = self.drone_camera.get_heightmap()

        if img is None or height_map is None:
            return None, None, None, None, 0, 0

        img_resized = img.resize(height_map.size, Image.Resampling.LANCZOS)
        height_map_rgb = height_map.convert('RGB')
        screen_blend = ImageChops.add(img_resized, height_map_rgb).convert('L')

        img_resized.convert("L").save(Path(__file__).parent.parent / "debug" / "base_scan.png")
        height_map.convert("L").save(Path(__file__).parent.parent / "debug" / "height_map.png")
        screen_blend.save(Path(__file__).parent.parent / "debug" / "scan_screen.png")

        w, h = height_map.size
        return screen_blend, img_resized, height_map, max_dist, w, h

    def _compute_world_points(self, corners, depth_array, width, height, eye, forward, right, up, tan_half_fov, aspect):
        points = []
        for (cx, cy) in corners:
            ix = int(round(cx))
            iy = int(round(cy))
            if ix < 0 or ix >= width or iy < 0 or iy >= height:
                continue
            d = depth_array[iy, ix]
            u = (2.0 * (cx + 0.5) / width - 1.0) * tan_half_fov * aspect
            v = (1.0 - 2.0 * (cy + 0.5) / height) * tan_half_fov
            direction = forward + right * u + up * v
            direction = direction / np.linalg.norm(direction)
            world_point = eye + d * direction
            points.append(world_point)
        return np.array(points) if len(points) == 4 else None

    def _compute_orientation(self, world_points):
        center = np.mean(world_points, axis=0)

        # Стороны
        v1 = world_points[1] - world_points[0]
        v2 = world_points[3] - world_points[0]

        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None, None, None, None, None
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal

        # Ось X вдоль v1, Y перпендикулярно
        X = v1 / np.linalg.norm(v1)
        Y = np.cross(normal, X)
        Y = Y / np.linalg.norm(Y)
        if np.dot(Y, v2) < 0:
            Y = -Y
        X = np.cross(Y, normal)
        X = X / np.linalg.norm(X)

        R = np.column_stack((X, Y, normal))  # матрица поворота (столбцы – локальные оси)

        # Извлечение углов Эйлера (ZYX)
        if abs(R[2, 0]) < 1.0 - 1e-6:
            ry = -np.arcsin(R[2, 0])
            cos_ry = np.cos(ry)
            rx = np.arctan2(R[2, 1] / cos_ry, R[2, 2] / cos_ry)
            rz = np.arctan2(R[1, 0] / cos_ry, R[0, 0] / cos_ry)
        else:
            rz = 0.0
            if R[2, 0] < 0:
                ry = np.pi / 2
                rx = np.arctan2(R[0, 1], R[0, 2])
            else:
                ry = -np.pi / 2
                rx = np.arctan2(-R[0, 1], -R[0, 2])

        return center, normal, rx, ry, rz

    def _align_drone(self, world_points):
        """Вычисляет целевую позицию и углы, сохраняет их для анимации."""
        if world_points is None or len(world_points) != 4:
            print("Недостаточно точек для выравнивания")
            return

        result = self._compute_orientation(world_points)
        if result[0] is None:
            print("Ошибка вычисления ориентации")
            return

        center, normal, rx, ry, rz = result

        # Позиция дрона: центр площадки + нормаль * (половина высоты дрона) + небольшой запас
        # Добавляем 0.1, чтобы точно не уйти под поверхность
        pos = center + normal * (Config.DRONE_H + 2 * Config.PLAT_H + 10)

        # Сохраняем цели для анимации
        self.landing_target_pos = pos.astype(np.float32)
        self.landing_target_rot = np.array([rx, ry, rz + np.pi / 2], dtype=np.float32)
        self.landing_progress = 0.0
        self.is_landing = True

    def _align_and_land(self):
        if self.is_landing:
            print("Уже выполняется посадка")
            return

        screen_blend, _, height_map, max_dist, w, h = self._get_scan_data()
        if screen_blend is None:
            print("Не удалось получить данные с камеры")
            return

        result = self.rect_processor.process(screen_blend, debug_name="landing_rect.png")
        if len(result.corners) == 0:
            print("Прямоугольник не обнаружен")
            return

        corners = result.corners
        depth_array = np.array(height_map) / 255.0 * max_dist

        cam = self.drone_camera
        eye = cam.get_eye()
        forward = cam.get_forward()
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        tan_half_fov = np.tan(cam.fov / 2.0)
        aspect = w / h

        world_points = self._compute_world_points(
            corners, depth_array, w, h,
            eye, forward, right, up, tan_half_fov, aspect
        )
        if world_points is None:
            print("Ошибка вычисления мировых координат")
            return

        self._align_drone(world_points)

    def update(self, dt):
        """Обновление состояния дрона (плавная посадка)."""
        if not self.is_landing:
            return

        self.landing_progress += dt / self.landing_duration
        if self.landing_progress >= 1.0:
            self.landing_progress = 1.0
            self.is_landing = False

        t = self.landing_progress

        # Интерполяция позиции
        current_pos = self.drone.position
        target_pos = self.landing_target_pos
        self.drone.position = (1 - t) * current_pos + t * target_pos

        # Интерполяция углов (линейная с учётом перехода через ±π)
        current_rot = self.drone.rotation
        target_rot = self.landing_target_rot
        delta = target_rot - current_rot
        # Нормализуем разность в диапазон [-π, π]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        self.drone.rotation = current_rot + delta * t

    def handle_events(self, dt):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False
                if event.key == pg.K_r:
                    self._align_and_land()
                if event.key == pg.K_p:          # новая клавиша
                    self._align_and_land()
            elif event.type == pg.MOUSEWHEEL:
                self.camera.fov += event.y * self.zoom_speed
                self.camera.fov = clamp(self.camera.fov, np.radians(10), np.radians(120))
            elif event.type == pg.MOUSEMOTION:
                dx, dy = event.rel
                self.camera.yaw += dx * self.mouse_sensitivity
                self.camera.pitch -= dy * self.mouse_sensitivity
                self.camera.pitch = clamp(self.camera.pitch, np.radians(-80), np.radians(80))

        if self.camera.attached_drone is None:
            keys = pg.key.get_pressed()
            forward = np.array([
                np.cos(self.camera.pitch) * np.sin(self.camera.yaw),
                np.cos(self.camera.pitch) * np.cos(self.camera.yaw),
                0.0
            ], dtype=np.float32)
            norm = np.linalg.norm(forward)
            if norm > 0:
                forward /= norm
            else:
                forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            right = np.array([-np.cos(self.camera.yaw), np.sin(self.camera.yaw), 0.0], dtype=np.float32)

            if keys[pg.K_w]:
                self.camera.eye_pos += forward * self.move_speed * dt
            if keys[pg.K_s]:
                self.camera.eye_pos -= forward * self.move_speed * dt
            if keys[pg.K_a]:
                self.camera.eye_pos += right * self.move_speed * dt
            if keys[pg.K_d]:
                self.camera.eye_pos -= right * self.move_speed * dt
            if keys[pg.K_SPACE]:
                self.camera.eye_pos[2] += self.move_speed * dt
            if keys[pg.K_LSHIFT]:
                self.camera.eye_pos[2] -= self.move_speed * dt

    def render(self):
        self.ctx.clear(0.1, 0.1, 0.15)
        self.camera.render()
        pg.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0
            self.handle_events(dt)
            self.update(dt)          # обновление анимации
            self.render()

        pg.mouse.set_visible(True)
        pg.event.set_grab(False)
        pg.quit()