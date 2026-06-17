from .utils import SingletonMeta
from pathlib import Path
from tomllib import load


_CONFIG_PATH = Path(__file__).parent.parent / "simulation_config.toml"
with open(_CONFIG_PATH, mode="rb") as config_file: _toml_data = load(config_file)

def get_val(section, key, default):
    data = _toml_data.get(section, {})
    val = data.get(key)
    return val if val is not None else default


class Config(metaclass=SingletonMeta):
    IS_SEEDED           = get_val("core", "IS_SEEDED", True)
    SEED                = get_val("core", "SEED", 0)
    SCALE               = get_val("core", "SCALE", 32)
    MAX_DIST_ERROR      = get_val("core", "MAX_DIST_ERROR", 16)

    FIELD_W             = get_val("field", "FIELD_W", 512)
    FIELD_L             = get_val("field", "FIELD_L", 512)
    OCTAVES             = get_val("field", "OCTAVES", 8)
    NOISE_SCALE         = get_val("field", "NOISE_SCALE", 4)
    STEP                = get_val("field", "STEP", 8)
    COMP_RATE           = get_val("field", "COMP_RATE", 1)
    BASE_MOD            = get_val("field", "BASE_MOD", 4)

    PLAT_W              = get_val("platform", "PLAT_W", 64)
    PLAT_L              = get_val("platform", "PLAT_L", 64)
    PLAT_H              = get_val("platform", "PLAT_H", 48)
    VERTEX_DISPLACEMENT = get_val("platform", "VERTEX_DISPLACEMENT", 16)
    PLAT_DISPLACEMENT   = get_val("platform", "PLAT_DISPLACEMENT", 96)

    DRONE_W             = get_val("drone", "DRONE_W", 48)
    DRONE_L             = get_val("drone", "DRONE_L", 48)
    DRONE_H             = get_val("drone", "DRONE_H", 16)
    INIT_ALTITUDE       = get_val("drone", "INIT_ALTITUDE", 500)

    PITCH               = get_val("camera", "PITCH", -90.0)
    YAW                 = get_val("camera", "YAW", 0.0)
    FOV_DEG             = get_val("camera", "FOV_DEG", 50.0)
    V_FOV_DEG           = get_val("camera", "V_FOV_DEG", 50.0)
    NEAR                = get_val("camera", "NEAR", 0.1)
    FAR                 = get_val("camera", "FAR", 10000.0)

    WINDOW_W            = get_val("render", "WINDOW_W", 1280)
    WINDOW_H            = get_val("render", "WINDOW_H", 720)
    FPS                 = get_val("render", "FPS", 30)

    RAY_STEP            = get_val("raytracing", "RAY_STEP", 4)
    MAX_RAY_DIST        = get_val("raytracing", "MAX_RAY_DIST", 1000)

    SPEED               = get_val("controls", "SPEED", 100)
    MOUSE_SENSITIVITY   = get_val("controls", "MOUSE_SENSITIVITY", 0.002)
    ZOOM_SPEED          = get_val("controls", "ZOOM_SPEED", 0.05)