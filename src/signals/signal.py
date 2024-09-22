from sklearn.preprocessing import MinMaxScaler

class Signal:
    def __init__(self, sensor:str, name: str, sampling: int, data: list[list[float]]):

        self.data = MinMaxScaler().fit_transform(data)
        self.sensor = sensor
        self.name = name
        self.sampling = sampling
        # self.data = data
