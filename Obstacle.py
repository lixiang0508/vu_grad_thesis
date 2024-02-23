class Obstacle:
    def __init__(self, weight, feature, points):
        self.weight = weight
        self.feature = feature # soft or hard
        self.points = points # corners
