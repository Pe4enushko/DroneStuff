from .config import Config
import numpy as np
import moderngl as mgl


class Drone:
    def __init__(self, ctx: mgl.Context):
        self.ctx = ctx

        self.position = np.array([Config.FIELD_W / 2, Config.FIELD_L / 2, Config.INIT_ALTITUDE], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.camera = None
        self.vertices, self.indices = self._generate_box()

        self.vbo = None
        self.ibo = None
        if self.ctx is not None: self.create_buffers()

    def create_buffers(self):
        if self.ctx is None: raise RuntimeError("Cannot create buffers without OpenGL context")
        self.vbo = self.ctx.buffer(self.vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(self.indices.astype('i4').tobytes())

    def _generate_box(self):
        w = Config.DRONE_W / 2
        l = Config.DRONE_L / 2
        h = Config.DRONE_H / 2

        vertices = np.array([
            [-w, -l, -h, 0.5, 0.5, 0.5],
            [w, -l, -h, 0.5, 0.5, 0.5],
            [w, l, -h, 0.5, 0.5, 0.5],
            [-w, l, -h, 0.5, 0.5, 0.5],
            [-w, -l, h, 0.7, 0.7, 0.7],
            [w, -l, h, 0.7, 0.7, 0.7],
            [w, l, h, 0.7, 0.7, 0.7],
            [-w, l, h, 0.7, 0.7, 0.7],
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,
            0, 2, 3,
            4, 6, 5,
            4, 7, 6,
            0, 5, 1,
            0, 4, 5,
            1, 6, 2,
            1, 5, 6,
            2, 7, 3,
            2, 6, 7,
            3, 4, 0,
            3, 7, 4,
        ], dtype=np.int32)

        return vertices, indices

    def change_position(self, dx=0, dy=0, dz=0, tilt_x=0, tilt_y=0, tilt_z=0, t=1.0):
        self.position[0] += dx * t
        self.position[1] += dy * t
        self.position[2] += dz * t

        self.rotation[0] += np.radians(tilt_x) * t
        self.rotation[1] += np.radians(tilt_y) * t
        self.rotation[2] += np.radians(tilt_z) * t

    def get_model_matrix(self):
        rx, ry, rz = self.rotation
        Rx = np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]], dtype=np.float32)
        Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]], dtype=np.float32)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        T = np.array([[1, 0, 0, self.position[0]], [0, 1, 0, self.position[1]], [0, 0, 1, self.position[2]],[0, 0, 0, 1]], dtype=np.float32)

        model = T @ Rz @ Ry @ Rx
        return model.T

    def attach_camera(self, camera):
        self.camera = camera
        camera.drone = self

    def get_camera_position(self):
        h = Config.DRONE_H / 2
        local_pos = np.array([0, 0, -h, 1], dtype=np.float32)
        model_row_major = self.get_model_matrix().T
        world_pos = model_row_major @ local_pos
        return world_pos[:3]

    def get_vao(self, program):
        vao = self.ctx.vertex_array(program, [(self.vbo, '3f 3f', 'in_position', 'in_color')], self.ibo)
        return vao