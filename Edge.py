import numpy as np
import matplotlib.pyplot as plt

class Edge:
    def __init__(self, value, grad):
        self.value = value;
        self.grad  = grad;
