
class DataObj:
    def __init__(self):
        self.data = None
        self.target = None
        self.id = None

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def set_target(self, target):
        self.target = target

    def get_target(self):
        return self.target

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id