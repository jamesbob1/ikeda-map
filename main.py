import numpy as np
import matplotlib.pyplot as plt
import numba as nb

u = 0.918


@nb.njit()
def next_ikeda_map(x, y):
    t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
    x = 1 + u * (x * np.cos(t) - y * np.sin(t))
    y = u * (x * np.sin(t) + y * np.cos(t))
    return x, y


def main():
    x, y = np.random.randn(2, 2000) * 10

    trajectories = [[x, y]]
    for _ in range(1000):
        x, y = next_ikeda_map(x, y)
        trajectories.append([x, y])

    trajectories = np.array(trajectories).transpose((2, 1, 0))
    for b in trajectories:
        plt.plot(b[0, :], b[1, :], color="w", alpha=0.1, linewidth=0.1)

    plt.gca().set_facecolor('k')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
