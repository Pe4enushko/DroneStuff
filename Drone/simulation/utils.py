class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def clamp(val, _min: int | float, _max: int | float) -> int | float:
    return max(min(_max, val), _min)