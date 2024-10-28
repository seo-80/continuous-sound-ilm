#%%
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
rng = np.random.default_rng()

DATA_DIR = os.path.dirname(__file__) +"/../data/"

def variational_bayes(data, K, max_iter=10000):
    # データの初期化
    N, D = data.shape
    
    # 事前分布のパラメータ設定
    mu_0 = np.zeros(D)  # 平均の事前分布の平均
    beta_0 = 1.0  # 精度の事前分布のパラメータ
    W_0 = np.eye(D)  # 精度行列の事前分布のパラメータ
    nu_0 = D + 2  # 自由度の事前分布のパラメータ
    
    # 責任度の初期化
    responsibilities = rng.dirichlet(np.ones(K), N)
    
    # 事後分布のパラメータの初期化
    mu = np.random.rand(K, D)
    beta = np.ones(K)
    W = np.array([np.eye(D) for _ in range(K)])
    nu = np.full(K, nu_0)
    pi = np.ones(K) / K  # 混合比率

    for iteration in range(max_iter):
        # Eステップ: 責任度の更新
        for k in range(K):
            responsibilities[:, k] = pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=beta[k] * np.linalg.inv(W[k]))
            responsibilities[:, k] /= responsibilities[:, k].sum(keepdims=True) + 1e-10

        # Mステップ: パラメータの更新
        Nk = responsibilities.sum(axis=0)
        for k in range(K):
            mu[k] = (beta_0 * mu_0 + Nk[k] * (responsibilities[:, k][:, np.newaxis] * data).sum(axis=0)) / (beta_0 + Nk[k])
            beta[k] = beta_0 + Nk[k]
            x_centered = data - mu[k]
            S = (responsibilities[:, k][:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', x_centered, x_centered)).sum(axis=0) + beta_0 * np.outer(mu_0, mu_0) + W_0
            W[k] = W_0 + S
            nu[k] = nu_0 + Nk[k]
            pi[k] = Nk[k] / N
        
        if np.all(mu == mu_0) and np.all(beta == beta_0) and np.all(W == W_0) and np.all(nu == nu_0):
            print("Converged at iteration", iteration)
            break

    return pi, mu, beta, W, nu, responsibilities


def iterated_leaning(simulation_times):
    data_size=100
    data_sizes = np.random.multinomial(data_size, [0.5, 0.5])
    data = np.vstack([np.random.randn(data_sizes[0], 2) + np.array([3, 3]), np.random.randn(data_sizes[1], 2) + np.array([-3, -3])])  # 100個の2次元データ
    K = 3  # クラスタ数
    ims = []
    fig, ax = plt.subplots()    
    for t in range(simulation_times):
        pi, mu, beta, W, nu, responsibilities = variational_bayes(data, K)
        data_sizes = np.random.multinomial(data_size, pi)
        # data = np.vstack([rng.normal(means[k], covariances[k], (data_sizes[k], 2)) for k in range(K)])
        data = np.vstack([rng.multivariate_normal(means[k], covariances[k], data_sizes[k]) for k in range(K)])
        # ax.clear()
        # for i in range(data_size):
        #     print(np.argmax(responsibilities[i]))
        #     ax.scatter(data[i, 0], data[i, 1], c=2, cmap='Set1')
        im = ax.scatter(data[:, 0], data[:, 1], c=np.argmax(responsibilities, axis=1), cmap='Set1')
        ims.append([im])
    print(type(fig))
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    print(len(ims))
    plt.show()
    # saves_ani = input("y to save the animation")
    # if saves_ani == "y":
    #     ani.save(DATA_DIR+"animation/output.gif", writer="imagemagick")


def test():
    data = np.vstack([np.random.randn(100, 2) + np.array([3, 3]), np.random.randn(100, 2) + np.array([-3, -3])])  # 100個の2次元データ

    # 変分ベイズ法の実行
    K = 4  # クラスタ数
    pi, mu, beta, W, nu, responsibilities = variational_bayes(data, K)
    means = mu
    covariances = np.array([np.linalg.inv(W[k] * beta[k]) for k in range(K)])

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=np.argmax(responsibilities, axis=1), cmap='Set1')
    print("混合比率:", pi)
    print("平均:", means)
    print("共分散行列:", covariances)
    plt.show()
if __name__ == "__main__":
    # iterated_leaning(100)
    test()




# %%
