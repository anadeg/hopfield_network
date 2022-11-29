import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# количество обучающих выборок
LEARNING_SAMPLES = 10


class HopfieldNetwork:
    def __init__(self, X_file):
        X = np.array(pd.read_csv(os.path.join("recognise", X_file), header=None))
        self.X = list(X)
        self.n = len(self.X)
        self.image_size = self.n * self.n
        self.W = [[0 for i in range(self.image_size)] for j in range(self.image_size)]
        self.known_digits = self.read_digits()

    def calculate_weights(self):
        pass

    # считывание известных цифр
    @staticmethod
    def read_digits():
        known_digits = []
        for i in range(LEARNING_SAMPLES):
            filename = f"digits - {i}.csv"
            digit = pd.read_csv(os.path.join("digits", filename), header=None)
            known_digits.append(HopfieldNetwork.reshape(digit, -1, 1))
        return known_digits

    # веса формируются по правилу дельта-проекций
    def update_weight(self, error, h):
        iter_num = 0
        run = True
        while run:
            iter_num += 1
            oldW = self.W[:]
            for i in range(LEARNING_SAMPLES):
                learning_matrix = self.known_digits[i]
                Xi = HopfieldNetwork.reshape(learning_matrix, -1, 1)

                alpha = (h / self.image_size)
                dW = self.multiply(self.dot_product(self.sub(Xi, self.dot_product(self.W, Xi)), self.transpose(Xi)), alpha)
                self.W = self.sub(self.W, self.multiply(dW, -1))
            newW = self.W[:]
            diff = self.abs_sum(self.sub(oldW, newW))
            # процесс "обучения" весов повторяется,
            # пока разность между новыми и старыми весами не будет меньше
            # заданной погрешности
            if iter_num % 50 == 0:
                print(f'iteration {iter_num}\t|\tdifference {diff}')
            if diff < error:
                run = False
                break

    @staticmethod
    def activate(x):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = np.tanh(x[i][j])
        return x

    # обучение сети
    # если енергия прекращает изменяться, сеть вошла в стабильное состояние
    def recognise(self, iters=np.inf):
        run = True
        learning_iters = 0
        x_reshaped = HopfieldNetwork.reshape(self.X, 1, -1)
        old = self.activate(self.dot_product(x_reshaped, self.W))
        while run and learning_iters < iters:
            learning_iters += 1
            temp_x = self.activate(self.dot_product(HopfieldNetwork.reshape(self.X, 1, -1), self.W))
            temp_x = self.activate(temp_x)
            self.X = temp_x
            new = self.activate(self.dot_product(HopfieldNetwork.reshape(self.X, 1, -1), self.W))
            if old == new:
                run = False
                print(f'iteration {learning_iters}')
            else:
                old = new
            if learning_iters % 5 == 0:
                print(f'iteration {learning_iters}')

        return HopfieldNetwork.reshape(self.activate(self.dot_product(HopfieldNetwork.reshape(self.X, 1, -1), self.W)), self.n, self.n)

    def fit(self, error, h):
        self.update_weight(error, h)

    @staticmethod
    def dot_product(x, y):
        i_x, j_y = len(x), len(y[0])
        result = [[0 for i in range(j_y)] for j in range(i_x)]
        inner = len(x[0])
        for i in range(i_x):
            for j in range(j_y):
                temp = 0
                for i_inner in range(inner):
                    temp += np.multiply(x[i][i_inner], y[i_inner][j])
                result[i][j] = temp
        return result


    @staticmethod
    def reshape(matrix, i, j):
        if i == -1 or i == 1:
            try:
                return [[matrix[i][j]] for i in range(len(matrix)) for j in range(len(matrix[0]))]
            except TypeError:
                return [[matrix[i]] for i in range(len(matrix))]
        if j == -1 or j == 1:
            try:
                return [matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0]))]
            except TypeError:
                return [matrix[i] for i in range(len(matrix))]
        else:
            remade = HopfieldNetwork.reshape(matrix, 1, -1)
            result = [[0 for j_i in range(j)] for i_i in range(i)]
            for i_temp in range(len(result)):
                for j_temp in range(len(result[i_temp])):
                    result[i_temp][j_temp] = remade[i_temp * j + j_temp]
            return result

    @staticmethod
    def transpose(x):
        h, w = len(x), len(x[0])
        transposed = [[x[j][i] for j in range(h)] for i in range(w)]
        return transposed

    @staticmethod
    def sub(a, b):
        x, y = len(a), len(a[0])
        result = [[a[i][j] - b[i][j] for j in range(y)] for i in range(x)]
        return result

    @staticmethod
    def multiply(a, coefficient):
        x, y = len(a), len(a[0])
        a = [[np.multiply(coefficient, a[i][j]) for j in range(y)] for i in range(x)]
        return a

    @staticmethod
    def abs_sum(x):
        result = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                result += x[i][j]
        return np.abs(result)

    def save(self, w_name, x_name):
        np.savetxt(w_name, np.array(self.W))
        np.savetxt(x_name, np.array(self.X))

    def read(self, w_name, x_name):
        self.W = list(np.loadtxt(w_name))
        X = np.loadtxt(x_name)
        self.X = [[X[i]] for i in range(X.shape[0])]


def show(X):
    plt.matshow(X)
    plt.show()


if __name__ == '__main__':
    file = "digits - sample2.csv"
    hn = HopfieldNetwork(file)
    show(hn.X)

    hn.read(os.path.join('savings', 'w2.txt'), os.path.join('savings', 'x2.txt'))
    # processed = hn.recognise(iters=0)
    hn.X, hn.W = np.array(hn.X), np.array(hn.W)

    # нумпай точнее работает с малыми числами
    processed = np.tanh(hn.X.reshape(1, -1) @ hn.W).reshape(hn.n, hn.n)
    show(processed)

