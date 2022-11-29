# сеть Хопфилда с непрерывным состоянием и дискретным временем в синхронном режиме
# режим распознавания (цифры в матрицах 7 на 7)
#
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# количество обучающих выборок
LEARNING_SAMPLES = 10


class HopfieldNetwork:
    def __init__(self, X_file):
        self.X = np.array(pd.read_csv(os.path.join("recognise", X_file), header=None))
        self.n = self.X.shape[0]
        self.image_size = self.n * self.n
        self.W = np.zeros((self.image_size, self.image_size))
        self.known_digits = np.array(self.read_digits())

    def calculate_weights(self):
        pass

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
        old = self.activate(self.X.reshape(1, -1) @ self.W)
        while run and learning_iters < iters:
            learning_iters += 1
            temp_x = self.activate(self.X.reshape(1, -1) @ self.W)
            self.X = temp_x
            new = self.activate(self.X.reshape(1, -1) @ self.W)
            if np.allclose(old, new):
                run = False
                print(f'iteration {learning_iters}')
            else:
                old = new
            if learning_iters % 1000 == 0:
                print(f'iteration {learning_iters}')
        return self.activate(self.X.reshape(1, -1) @ self.W).reshape(self.n, self.n)

    def fit(self, error, h):
        self.update_weight(error, h)


def show(X):
    plt.matshow(X)
    plt.show()


if __name__ == '__main__':
    file = "digits - sample1.csv"
    hn = HopfieldNetwork(file)
    show(hn.X)

    hn.fit(0.0001, 0.9)
    print('=======================================================')
    processed = hn.recognise()
    show(processed)

