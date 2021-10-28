import numpy as np


def generate_2d_samples(x_max: int, y_max: int, size: int):
    x = np.random.randint(x_max, size=size)
    y = np.random.randint(y_max, size=size)
    return list(zip(x, y))


if __name__ == "__main__":
    print(generate_2d_samples(60, 60, 10))
