from .config import Config
import moderngl as mgl
import numpy as np
from pathlib import Path
from PIL import Image


class GPURaytracing:
    def __init__(self, camera, ctx: mgl.Context):
        self.camera = camera
        self.ctx = ctx
        path = Path(__file__).parent / "shaders" / "raytracing.comp"
        self.prog = self.ctx.compute_shader(path.read_text())

    def _get_vec4_buffer(self, vertices):
        pos = vertices[:, :3]
        buf = np.zeros((pos.shape[0], 4), dtype='f4')
        buf[:, :3] = pos
        return self.ctx.buffer(buf.tobytes())

    def capture(self) -> Image.Image:
        sw, sh = Config.WINDOW_W, Config.WINDOW_H
        step = Config.RAY_STEP
        w, h = sw // step, sh // step

        tex = self.ctx.texture((w, h), 4, dtype='f1')
        tex.bind_to_image(0, read=False, write=True)

        field = self.camera.field
        b_tv = self._get_vec4_buffer(field.vertices)
        b_ti = self.ctx.buffer(field.indices.astype('u4').tobytes())
        b_pv = self._get_vec4_buffer(field.platform_vertices)
        b_pi = self.ctx.buffer(field.platform_indices.astype('u4').tobytes())

        b_tv.bind_to_storage_buffer(0)
        b_ti.bind_to_storage_buffer(1)
        b_pv.bind_to_storage_buffer(2)
        b_pi.bind_to_storage_buffer(3)

        forward = self.camera.get_forward()

        world_up = np.array([0, 0, 1], dtype='f4')
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0], dtype='f4')
        else:
            right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        self.prog['cam_pos'].value = tuple(self.camera.get_eye())
        self.prog['cam_forward'].value = tuple(forward)
        self.prog['cam_right'].value = tuple(right)
        self.prog['cam_up'].value = tuple(up)

        self.prog['tan_half_fov'].value = np.tan(self.camera.fov / 2.0)
        self.prog['aspect_ratio'].value = sw / sh

        m_dist = max(float(Config.MAX_RAY_DIST), self.camera.get_eye()[2] + 200.0)
        self.prog['max_dist'].value = m_dist

        self.prog['terrain_tri_count'].value = len(field.indices) // 3
        self.prog['platform_tri_count'].value = len(field.platform_indices) // 3

        self.prog.run((w + 15) // 16, (h + 15) // 16)

        img_data = tex.read()
        img = Image.frombytes('RGBA', (w, h), img_data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM).convert('L')

        for r in [tex, b_tv, b_ti, b_pv, b_pi]:
            r.release()

        return img, m_dist