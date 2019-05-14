import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):

    def __init__(self, dimInput, dimOutput, func_act, func_act_deriv):
        self.w_ = np.random.rand(1 + dimInput, dimOutput) # random values between 0 and 1
        self.f_act = func_act
        self.f_act_deriv = func_act_deriv

    def train(self, X, y, eta=0.01, epochs=100):
        self.eta = eta
        self.epochs = epochs
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0

            for xi, target in zip(X, y):
                error = target - self.f_act(xi)

                update = self.eta * np.multiply(error, self.f_act_deriv(xi))

                self.w_[1:] += np.transpose(np.outer(update, xi))
                self.w_[0] += update * 1

                errors += int(update != 0.0)

            self.errors_.append(errors)

            if errors <= 0.01:
                break

        return self