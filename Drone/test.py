from simulation.field import Field
from simulation.drone import Drone
from simulation.camera import Camera
from simulation.renderer import Renderer
from simulation.config import Config


class Simulation:
    def __init__(self):
        self.config = Config()
        self.ctx = None
        self.field = Field(self.ctx)
        self.drone = Drone(self.ctx)
        self.free_camera = Camera(
            ctx=self.ctx,
            pos=[-200, -200, 500],
            pitch=-15, yaw=45.0,
            field=self.field,
            drone=self.drone
        )
        self.drone_camera = Camera(
            ctx=self.ctx,
            pitch=Config.PITCH, yaw=Config.YAW,
            field=self.field,
            at_drone=self.drone
        )
        self.drone.attach_camera(self.drone_camera)

        self.renderer = Renderer(self.field, self.drone, self.free_camera, self.drone_camera)

    def move_drone(self, dx=0, dy=0, dz=0, tilt_x=0, tilt_y=0, tilt_z=0, t=1.0):
        self.drone.change_position(dx, dy, dz, tilt_x, tilt_y, tilt_z, t)

    def run(self):
        self.renderer.run()

if __name__ == "__main__":
    import sys

    try:
        sim = Simulation()
        sim.run()
    except KeyboardInterrupt: sys.exit(0x0)
