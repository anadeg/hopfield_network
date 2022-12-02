# сеть Хопфилда с непрерывным состоянием и дискретным временем в синхронном режиме
# режим распознавания (цифры в матрицах 7 на 7)
#
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import time


# количество обучающих выборок
LEARNING_SAMPLES = 10


class HopfieldNetwork:
    def __init__(self, X_file):
        self.X = np.array(pd.read_csv(os.path.join("recognise", X_file), header=None))
        self.n = self.X.shape[0]
        self.image_size = self.n * self.n
        self.W = np.zeros((self.image_size, self.image_size))
        self.known_digits = np.array(self.read_digits())

    # считывание известных цифр
    @staticmethod
    def read_digits():
        known_digits = []
        for i in range(LEARNING_SAMPLES):
            filename = f"digits - {i}.csv"
            digit = pd.read_csv(os.path.join("digits", filename), header=None)
            known_digits.append(np.array(digit).reshape(-1, 1))
        return known_digits

    # веса формируются по правилу дельта-проекций
    def update_weight(self, error, h):
        iter_num = 0
        run = True
        start = time()
        while run:
            iter_num += 1
            oldW = self.W.copy()
            for i in range(self.known_digits.shape[0]):
                # берется одна обучающая выборка (цифра)
                learning_matrix = self.known_digits[i]
                Xi = learning_matrix.reshape(-1, 1)

                # применяется правило
                # W = W + nu/N * (Xi - W*Xi) * Xi.T
                # nu - константа в пределах [0.7, 0.9]
                # N - размер изображения
                dW = (h / self.image_size) * (Xi - self.W @ Xi) @ Xi.T
                self.W += dW
            newW = self.W.copy()
            diff = np.abs((oldW - newW).sum())
            # процесс "обучения" весов повторяется,
            # пока разность между новыми и старыми весами не будет меньше
            # заданной погрешности
            if iter_num % 50 == 0:
                print(f'iteration {iter_num}\t|\tdifference {diff}')
            if diff < error:
                run = False
                end = time()
                print(end - start, 'ms')
                break

    @staticmethod
    def activate(x):
        return np.tanh(x)

    def energy(self, x):
        return -0.5 * np.sum(x.T @ self.W @ x)

    # обучение сети
    # если енергия прекращает изменяться, сеть вошла в стабильное состояние
    def recognise(self, iters=np.inf):
        run = True
        learning_iters = 0
        old_2 = self.X
        old_1 = self.activate(self.X.reshape(1, -1) @ self.W)
        while run and learning_iters < iters:
            learning_iters += 1
            temp_x = self.activate(self.X.reshape(1, -1) @ self.W)
            self.X = temp_x
            new = self.activate(self.X.reshape(1, -1) @ self.W)
            if np.allclose(old_2.reshape(-1, self.image_size), new):
                run = False
                print(f'iteration {learning_iters}')
            else:
                old_2 = old_1
                old_1 = new
            if learning_iters % 1000 == 0:
                print(f'iteration {learning_iters}')
        return self.activate(self.X.reshape(1, -1) @ self.W).reshape(self.n, self.n)

    def fit(self, error, h):
        self.update_weight(error, h)

    def save(self, w_name, x_name):
        np.savetxt(w_name, self.W)
        np.savetxt(x_name, self.X)

    def read(self, w_name, x_name):
        self.W = np.loadtxt(w_name)
        self.X = np.loadtxt(x_name)


def show(X):
    plt.matshow(X)
    plt.show()


if __name__ == '__main__':
    file = "digits - sample2.csv"
    hn = HopfieldNetwork(file)
    show(hn.X)

    hn.fit(0.00001, 0.9)
    print('=======================================================')
    processed = hn.recognise()
    # hn.save(os.path.join('savings', 'w2.txt'), os.path.join('savings', 'x2.txt'))
    show(processed)

