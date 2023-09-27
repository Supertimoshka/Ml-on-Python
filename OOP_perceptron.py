import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

figures = [
    {'x': -2, 'y': 2, 'type': 1}, # квадрат = 1, круг = 0
    {'x': -3, 'y': 6, 'type': 1},
    {'x': -4, 'y': 4, 'type': 1},
    {'x': 1, 'y': -2, 'type': 0},
    {'x': 2, 'y': 1, 'type': 0},
    {'x': 3, 'y': 2, 'type': 0},
    {'x': -1, 'y': 2, 'type': 1},
    {'x': 0, 'y': 6, 'type': 0}
]


class Neuron:

    def __init__(self):
        self.w = np.array([random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)])

    def activate(self, input_vector):
        scalar_product = np.dot(input_vector, self.w)
        return Neuron.sigmoid(scalar_product)

    def sigmoid(scalar_product):
        return 1 / (1 + np.exp(scalar_product))
    

class Perceptron:

    def __init__(self, epochs_number = 1000, learning_rate = 0.5):
        self.epochs_number = epochs_number
        self.learning_rate = learning_rate
        self.neuron = Neuron()

    def learn(self, figures):
        for i in range(0, self.epochs_number):
            self.run_step_learning(figures)

    def draw_result_learning(self, figures):

        coeff_k = - self.neuron.w[1] / self.neuron.w[2]
        coeff_b = - self.neuron.w[0] / self.neuron.w[2]
        # print(coeff_k)
        # print(coeff_b)

        graph = plt.figure(figsize = (6, 6))

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.grid()

        x = np.linspace(-7, 7, 1000)
        y = coeff_k * x + coeff_b

        plt.plot(x, y)
        axes = plt.gca()

        for figure in figures:
            if figure['type'] == 1:
                rectangle = matplotlib.patches.Rectangle((figure['x'], figure['y']), 0.2, 0.2, color='b')
                axes.add_patch(rectangle)
            else:
                circle = matplotlib.patches.Circle((figure['x'], figure['y']), 0.1, color='r')
                axes.add_patch(circle)

        plt.show()

    def run_step_learning(self, figures):
        figure = figures[random.randint(0, len(figures) - 1)]
        output = self.run_forward_propagation(figure)
        self.run_back_propagation(output, figure)
        
    def run_forward_propagation(self, figure):
        input_vector = np.array([1, figure['x'], figure['y']])
        return self.neuron.activate(input_vector)
    
    def run_back_propagation(self, output, figure):
        input_vector = np.array([1, figure['x'], figure['y']])
        self.neuron.w -= self.learning_rate * (output - figure['type']) * (-1) * output * (1 - output) * input_vector


p = Perceptron()
p.learn(figures)
p.draw_result_learning(figures)