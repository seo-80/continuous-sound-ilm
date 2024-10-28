import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import numpy as np
import os
import json


#load data

DATA_DIR = os.path.dirname(__file__) +"/../data/"
# 合成データを生成
np.random.seed(0)
K = 4  # 混合成分数
true_K = 4
D = 2  # 次元数
N = 1000  # サンプル数
filter_name = "high_entropy"
filter_name = "none"
# 真の混合ガウス分布のパラメータ
true_alpha = np.array([1/true_K, 1/true_K, 1/true_K, 1/true_K])
true_means = np.array([[5, 5], [-5, 5], [5, -5], [-5, -5]])
true_covars = np.array([[[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],])
true_covars = np.array([np.eye(D)*50 for _ in range(true_K)])

# モデルを初期化
alpha0 = 100.0
# alpha0 = 1.0
c_alpha = 0.1
beta0 = 1.0
nu0 = D + 2.0
m0 = np.zeros(D)
m0 = np.array([[5, 5], [-5, 5], [5, -5], [-5, -5]])
# m0 = np.array([[0, 0], [-5, 5], [5, -5], [-5, -5]])
W0 = np.eye(D)*0.02

iter = 100
agent = "BayesianGaussianMixtureModelWithContext"
# agent = "BayesianGaussianMixtureModel"
config = {
    "K": K,
    "D": D,
    "N": N,
    "c_alpha": c_alpha,
    "agent": agent,
    'true_alpha': true_alpha.tolist(),
    'true_means': true_means.tolist(),
    'true_covars': true_covars.tolist(),
    "alpha0": alpha0,
    "beta0": beta0,
    "nu0": nu0,
    "m0": m0.tolist(),
    "W0": W0.tolist(),
    "iter": iter,   
    "filter_func": filter_name,
    "filter_args": {
        "threshold": 0.125
    }
}



folder_names = os.listdir(DATA_DIR)
def dict_equal(d1, d2):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if len(d1) != len(d2):
            return False
        for k, v1 in d1.items():
            if k not in d2:
                return False
            v2 = d2[k]
            if not dict_equal(v1, v2):
                return False
        return True
    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            return False
        for v1, v2 in zip(d1, d2):
            if not dict_equal(v1, v2):
                return False
        return True
    elif isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
        return np.array_equal(d1, d2)
    else:
        return d1 == d2
print(config)
for folder_name in folder_names:
    if os.path.exists(DATA_DIR+folder_name+"/config.json"):
        with open(DATA_DIR+folder_name+"/config.json") as f:
            temp_config = json.load(f)
        print(temp_config)
        if dict_equal(temp_config, config):
            X = np.load(DATA_DIR+folder_name+"/data.npy")
            params = np.load(DATA_DIR+folder_name+"/params.npy", allow_pickle=True).item()
            retry_counts = np.load(DATA_DIR+folder_name+"/retry_counts.npy")
            print("load", folder_name)
            break
    
if "X" not in locals():
    print("Not Found")
    exit()


# fig, axs = plt.subplots()
# axs.plot(retry_counts)
# plt.show()
# fig, axs = plt.subplots()
# # Calculate the distance from the center for each cluster

# Plot the trajectory of params["m"] in 2D space
fig, axs = plt.subplots()
# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, iter))

# Plot the trajectory of params["m"] in 2D space with color gradient
for k in range(K):
    for i in range(1, iter):
        axs.plot(params["m"][i-1:i+1, k, 0], params["m"][i-1:i+1, k, 1], color=colors[i], alpha=0.5)
    axs.scatter(params["m"][-1, k, 0], params["m"][-1, k, 1], color='b', marker='o', s=10, label=f"Cluster {k+1}")

# for k in range(K):
#     axs.plot(params["m"][:, k, 0], params["m"][:, k, 1], label=f"Cluster {k+1}")
axs.set_xlim(-10, 10)
axs.set_ylim(-10, 10)
axs.set_xlabel("X")
axs.set_ylabel("Y")
axs.legend()
plt.savefig(os.path.join(DATA_DIR, folder_name,"trajectory.png"))
plt.close(fig)

# plt.show()


# # Plot the distances
fig, axs = plt.subplots()
distances = np.sqrt(np.sum((params["m"] - np.mean(params["m"], axis=1, keepdims=True))**2, axis=2))
for k in range(K):
    axs.plot(distances[:, k], label=f"Cluster {k+1}")
axs.set_xlabel("Iteration")
axs.set_ylabel("Distance from Center")
axs.legend()
plt.savefig(os.path.join(DATA_DIR, folder_name,"distance.png"))
# plt.show()


# Plot the trajectory of params["m"] in 2D space with color gradient
fig, axs = plt.subplots()
def update(i):
    axs.clear()
    artists = []

    scatter = axs.scatter(X[i][:, 0], X[i][:, 1], s=5, alpha=0.5)
    artists.append(scatter)

    for k in range(K):
        mean = params["m"][i, k]
        matrix = params["beta"][i, k] * params["W"][i, k, :, :]
        covar = np.linalg.inv(matrix)
        axs.set_xlim(-10, 10)
        axs.set_ylim(-10, 10)
        x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        xy = np.column_stack([x.flat, y.flat])
        z = multivariate_normal.pdf(xy, mean=mean, cov=covar).reshape(x.shape)

        rv = multivariate_normal(mean, covar)
        level = rv.pdf(mean) * np.exp(-0.5 * (np.sqrt(2)) ** 2)
        contour = axs.contour(x, y, z, alpha=0.5, levels=[level])
        artists.append(contour)

    # axs.set_title(f"iteration {i}")

    # plt.tight_layout()

    return artists

ani = animation.FuncAnimation(fig, update, frames=iter, interval=500, blit=True)
ani.save(DATA_DIR+folder_name+"/animation.gif", writer="pillow")
# plt.show()
