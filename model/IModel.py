from interface import Interface


class IModel(Interface):

    def evaluate(self, images, labels):
        pass

    def train(self, images, labels):
        pass
