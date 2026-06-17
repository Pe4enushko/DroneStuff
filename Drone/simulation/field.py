from .config import Config
import numpy as np
import moderngl as mgl
from noise import pnoise2


class Field:
    def __init__(self, ctx: mgl.Context):
        self.ctx = ctx
        self.config = Config()

        if self.config.IS_SEEDED: np.random.seed(self.config.SEED)

        self.vertices, self.indices = self._generate_terrain()
        self.platform_vertices, self.platform_indices = self._generate_platform()

        self.vbo = None
        self.ibo = None
        self.platform_vbo = None
        self.platform_ibo = None

        if self.ctx is not None: self.create_buffers()

    def create_buffers(self):
        if self.ctx is None: raise RuntimeError("Cannot create buffers without OpenGL context")
        self.vbo = self.ctx.buffer(self.vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(self.indices.astype('i4').tobytes())
        self.platform_vbo = self.ctx.buffer(self.platform_vertices.astype('f4').tobytes())
        self.platform_ibo = self.ctx.buffer(self.platform_indices.astype('i4').tobytes())

    def _generate_terrain(self):
        w, l = self.config.FIELD_W, self.config.FIELD_L
        step = self.config.STEP
        scale = self.config.SCALE
        comp_rate = self.config.COMP_RATE

        grid_w = w // step + 1
        grid_l = l // step + 1

        vertices = []

        for i in range(grid_l):
            for j in range(grid_w):
                x = j * step
                y = i * step

                noise_val = pnoise2(x / self.config.NOISE_SCALE / scale, y / self.config.NOISE_SCALE / scale, octaves=self.config.OCTAVES, repeatx=1024, repeaty=1024)
                noise_val = (noise_val + 1) / 2
                z = noise_val * scale / comp_rate
                color = noise_val
                color = np.clip(z * Config.BASE_MOD, 0, 255.0) / 255.0
                vertices.append([x, y, z, color, color, color])

        vertices = np.array(vertices, dtype=np.float32)

        indices = []
        for i in range(grid_l - 1):
            for j in range(grid_w - 1):
                idx = i * grid_w + j

                indices.extend([idx, idx + grid_w, idx + 1])
                indices.extend([idx + 1, idx + grid_w, idx + grid_w + 1])

        indices = np.array(indices, dtype=np.int32)

        return vertices, indices

    def _generate_platform(self):
        w = self.config.PLAT_W
        l = self.config.PLAT_L
        h = self.config.PLAT_H
        vertex_disp = self.config.VERTEX_DISPLACEMENT
        plat_disp = self.config.PLAT_DISPLACEMENT

        center_x = self.config.FIELD_W / 2
        center_y = self.config.FIELD_L / 2

        displacement = np.random.uniform(0, plat_disp)
        angle = np.random.uniform(0, 2 * np.pi)

        offset_x = center_x + displacement * np.cos(angle)
        offset_y = center_y + displacement * np.sin(angle)

        rotation = np.random.uniform(0, 2 * np.pi)

        base_vertices = [
            [-w / 2, -l / 2, 0],
            [w / 2, -l / 2, 0],
            [w / 2, l / 2, 0],
            [-w / 2, l / 2, 0],
        ]

        top_vertices = [
            [-w / 2, -l / 2, h + np.random.uniform(-vertex_disp, vertex_disp)],
            [w / 2, -l / 2, h + np.random.uniform(-vertex_disp, vertex_disp)],
            [w / 2, l / 2, h + np.random.uniform(-vertex_disp, vertex_disp)],
        ]

        vert_4 = [a + b - c for a, b, c in zip(top_vertices[0], top_vertices[2], top_vertices[1])]
        top_vertices.append(vert_4)

        all_vertices = base_vertices + top_vertices

        vertices_transformed = []
        for v in all_vertices:
            x, y, z = v
            x_rot = x * np.cos(rotation) - y * np.sin(rotation)
            y_rot = x * np.sin(rotation) + y * np.cos(rotation)
            x_final = x_rot + offset_x
            y_final = y_rot + offset_y
            z_final = z

            color = z / (h + vertex_disp) if (h + vertex_disp) > 0 else 0
            color = np.clip(color, 0, 1)
            color = np.clip(z * Config.BASE_MOD, 0, 255.0) / 255.0

            vertices_transformed.append([x_final, y_final, z_final, color, color, color])

        vertices = np.array(vertices_transformed, dtype=np.float32)

        indices = [
            4, 5, 6,
            4, 6, 7,
            0, 1, 5,
            0, 5, 4,
            1, 2, 6,
            1, 6, 5,
            2, 3, 7,
            2, 7, 6,
            3, 0, 4,
            3, 4, 7,
        ]

        indices = np.array(indices, dtype=np.int32)

        return vertices, indices

    def get_terrain_vao(self, program):
        vao = self.ctx.vertex_array(program, [(self.vbo, '3f 3f', 'in_position', 'in_color')], self.ibo)
        return vao

    def get_platform_vao(self, program):
        vao = self.ctx.vertex_array(program, [(self.platform_vbo, '3f 3f', 'in_position', 'in_color')], self.platform_ibo)
        return vao