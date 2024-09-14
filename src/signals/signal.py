class Signal:
    def __init__(self, sensor:str, name: str, sampling: int, data: list[float]):
        self.sensor = sensor
        self.name = name
        self.sampling = sampling
        self.data = data
