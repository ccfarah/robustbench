from cgi import test
from dataclasses import dataclass
from pyexpat import model


class ModelTester:
    def __init__(self, model, data, test):
        self.model = model
        self.data = data
        self.test = test

