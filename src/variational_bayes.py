import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def variational_bayes(data, K, max_iter=10000):
    # データの初期化
    N, D = data.shape
    # 責任度の初期化
    responsibilities = rng.dirichlet(np.ones(K), N)
    # パラメータの初期化
    means = np.random.rand(K, D)
    means_old = np.empty_like(means)
    covariances = np.array([np.eye(D) for _ in range(K)])
    pi = np.ones(K) / K  # 混合比率

    for iteration in range(max_iter):
        means_old=means.copy()
        # Eステップ: 責任度の更新
        for k in range(K):
            responsibilities[:, k] = pi[k] * multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # Mステップ: パラメータの更新
        Nk = responsibilities.sum(axis=0)
        for k in range(K):
            means[k] = (responsibilities[:, k][:, np.newaxis] * data).sum(axis=0) / Nk[k]
            x_centered = data - means[k]
            # covariances[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', x_centered, x_centered)).sum(axis=0) / Nk[k]
            covariances[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', x_centered, x_centered)).sum(axis=0) / Nk[k] + np.eye(D) * 1e-6
            pi[k] = Nk[k] / N
        if np.all(means_old == means):
            print("break", iteration)
            break

    return pi, means, covariances, responsibilities

def iterated_leaning(simulation_times):
    data_size=100
    data_sizes = np.random.multinomial(data_size, [0.5, 0.5])
    data = np.vstack([np.random.randn(data_sizes[0], 2) + np.array([3, 3]), np.random.randn(data_sizes[1], 2) + np.array([-3, -3])])  # 100個の2次元データ
    K = 3  # クラスタ数
    fig, ax = plt.subplots()
    for t in range(simulation_times):
        pi, means, covariances, responsibilities = variational_bayes(data, K)
        data_sizes = np.random.multinomial(data_size, pi)
        # data = np.vstack([rng.normal(means[k], covariances[k], (data_sizes[k], 2)) for k in range(K)])
        data = np.vstack([rng.multivariate_normal(means[k], covariances[k], data_sizes[k]) for k in range(K)])
        ax.clear()
        # for i in range(data_size):
        #     print(np.argmax(responsibilities[i]))
        #     ax.scatter(data[i, 0], data[i, 1], c=2, cmap='Set1')
        ax.scatter(data[:, 0], data[:, 1], c=np.argmax(responsibilities, axis=1), cmap='Set1')
        plt.pause(1)


def test():
    data = np.vstack([np.random.randn(100, 2) + np.array([3, 3]), np.random.randn(100, 2) + np.array([-3, -3])])  # 100個の2次元データ

    # 変分ベイズ法の実行
    K = 4  # クラスタ数
    pi, means, covariances, responsibilities = variational_bayes(data, K)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=np.argmax(responsibilities, axis=1), cmap='Set1')
    plt.show()
    print("混合比率:", pi)
    print("平均:", means)
    print("共分散行列:", covariances)
if __name__ == "__main__":
    iterated_leaning(100)
    # test()

