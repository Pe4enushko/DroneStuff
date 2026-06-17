from .config import Config
import numpy as np
import moderngl as mgl
from pathlib import Path
from PIL import Image


class Camera:
    def __init__(self, /, ctx: mgl.Context, pitch: float, yaw: float, field, drone = None, at_drone=None, pos: list = None):
        self.ctx = ctx
        self.field = field
        self.drone = drone
        self.attached_drone = at_drone

        self.pitch = np.radians(pitch)
        self.yaw = np.radians(yaw)
        self.fov = np.radians(Config.FOV_DEG)
        self.v_fov = np.radians(Config.V_FOV_DEG)
        self.eye_pos = pos

        self.program = None
        if self.ctx is not None: self.create_shaders()

        self.fbo = None

        self._terrain_vao = None
        self._platform_vao = None
        self._drone_vao = None

    def create_shaders(self):
        with open(Path(__file__).parent / "shaders" / "shader.vert", mode="r") as glsl_vert, \
            open(Path(__file__).parent / "shaders" / "shader.frag", mode="r") as glsl_frag:

            self.program = self.ctx.program(vertex_shader=glsl_vert.read(), fragment_shader=glsl_frag.read())

    def get_view_matrix(self):
        eye = self.get_eye()
        forward = self.get_forward()

        up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(forward, up)) > 0.999:
            up = np.array([0, 1, 0], dtype=np.float32)

        s = np.cross(forward, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, forward)

        res = np.array([
            [s[0], u[0], -forward[0], 0],
            [s[1], u[1], -forward[1], 0],
            [s[2], u[2], -forward[2], 0],
            [-np.dot(s, eye), -np.dot(u, eye), np.dot(forward, eye), 1]
        ], dtype=np.float32)

        return res

    def get_projection_matrix(self):
        aspect = Config.WINDOW_W / Config.WINDOW_H
        near = Config.NEAR
        far = Config.FAR
        f = 1.0 / np.tan(self.fov / 2)

        res = np.zeros((4, 4), dtype=np.float32)
        res[0, 0] = f / aspect
        res[1, 1] = f
        res[2, 2] = (far + near) / (near - far)
        res[2, 3] = -1
        res[3, 2] = (2 * far * near) / (near - far)
        return res

    def render(self):
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.15)

        view = np.ascontiguousarray(self.get_view_matrix().astype('f4'))
        projection = np.ascontiguousarray(self.get_projection_matrix().astype('f4'))
        model = np.ascontiguousarray(np.eye(4, dtype=np.float32))

        self.program['projection'].write(projection)
        self.program['view'].write(view)
        self.program['model'].write(model)

        if self._terrain_vao is None:
            self._terrain_vao = self.field.get_terrain_vao(self.program)
        self._terrain_vao.render(mgl.TRIANGLES)

        if self._platform_vao is None:
            self._platform_vao = self.field.get_platform_vao(self.program)
        self._platform_vao.render(mgl.TRIANGLES)

        if self.drone:
            model_drone = np.ascontiguousarray(self.drone.get_model_matrix().astype('f4'))
            self.program['model'].write(model_drone)

            if self._drone_vao is None: self._drone_vao = self.drone.get_vao(self.program)
            self._drone_vao.render(mgl.TRIANGLES)

    def get_scan(self) -> Image.Image | None:
        if self.attached_drone is None: return None

        if self.fbo is None:
            width, height = Config.WINDOW_W, Config.WINDOW_H
            color_tex = self.ctx.texture((width, height), 3)
            depth_rbo = self.ctx.depth_renderbuffer((width, height))
            self.fbo = self.ctx.framebuffer(color_tex, depth_rbo)

        self.fbo.use()
        self.ctx.viewport = (0, 0, Config.WINDOW_W, Config.WINDOW_H)
        self.render()

        data = self.fbo.read(components=3)
        img = Image.frombytes('RGB', self.fbo.size, data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, Config.WINDOW_W, Config.WINDOW_H)

        return img

    def get_eye(self) -> np.ndarray:
        if self.attached_drone is None: return np.array(self.eye_pos, dtype=np.float32)
        else: return self.attached_drone.get_camera_position()

    def get_forward(self) -> np.ndarray:
        forward = np.array([
            np.cos(self.pitch) * np.sin(self.yaw),
            np.cos(self.pitch) * np.cos(self.yaw),
            np.sin(self.pitch)
        ], dtype=np.float32)
        return forward / np.linalg.norm(forward)

    def get_heightmap(self):
        from .gpu_raytracer import GPURaytracing
        if self.attached_drone is None:
            return None, None
        raytracer = GPURaytracing(self, self.ctx)
        return raytracer.capture()