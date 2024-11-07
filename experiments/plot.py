import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import numpy as np
import os
import json
import xarray as xr
import sys
import argparse
from pathlib import Path






sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import metrics

parser = argparse.ArgumentParser(description='Process some data.')

parser.add_argument('--folder_name', type=str, default=None, help='input file path')


folder_name = parser.parse_args().folder_name


#load data

DATA_DIR = os.path.dirname(__file__) +"/../data/"
# 合成データを生成
np.random.seed(0)
K = 4  # 混合成分数
true_K = 4
D = 2  # 次元数
N = 1000  # サンプル数
filter_name = "high_entropy"
filter_name = "low_max_prob"
# filter_name = "none"
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
c_alpha = 0.1 # if c_alpha is None, agent inferes alpha and use it to generate data
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
        "threshold": 1-1/16
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
matched_data = []   
if folder_name:
    folder_names = [folder_name]

for folder_name in folder_names:
    if os.path.exists(DATA_DIR+folder_name+"/config.json"):
        with open(DATA_DIR+folder_name+"/config.json") as f:
            temp_config = json.load(f)
        X = np.load(DATA_DIR+folder_name+"/data.npy")
        Z = np.load(DATA_DIR+folder_name+"/Z.npy")
        C = np.load(DATA_DIR+folder_name+"/context.npy")
        params = np.load(DATA_DIR+folder_name+"/params.npy", allow_pickle=True).item()
        retry_counts = np.load(DATA_DIR+folder_name+"/retry_counts.npy")
        metrics_data = xr.open_dataset(os.path.join(DATA_DIR, folder_name, "metrics.nc"))
        matched_data.append({
            "X": X,
            "Z": Z,
            "C": C,
            "params": params,
            "retry_counts": retry_counts,
            "folder_name": folder_name,
            "metrics_data": metrics_data
        })
        print("load", folder_name)
            
    
if "X" not in locals():
    print("Not Found")
    exit()

for data in matched_data:
    X = data["X"]
    Z = data["Z"]
    C = data["C"]
    params = data["params"]
    retry_counts = data["retry_counts"]
    folder_name = data["folder_name"]
    metrics_data = data["metrics_data"]
    iter = params["m"].shape[0]

    # plot metrics
    ## plot expected_mahalanobis mean
    fig, axs = plt.subplots()
    expected_mahalanobis_mean = metrics_data['expected_mahalanobis'].mean(dim='simulation')
    im = axs.imshow(expected_mahalanobis_mean, cmap='viridis', origin='lower')
    axs.set_xticks(range(expected_mahalanobis_mean.shape[1]))
    axs.set_yticks(range(expected_mahalanobis_mean.shape[0]))
    axs.set_xticklabels(range(1, expected_mahalanobis_mean.shape[1]+1))
    axs.set_yticklabels(range(1, expected_mahalanobis_mean.shape[0]+1))
    axs.set_xlabel('Component')
    axs.set_ylabel('Component')
    axs.invert_yaxis()
    cbar = fig.colorbar(im)
    cbar.set_label('Expected Mahalanobis Distance')
    plt.savefig(os.path.join(DATA_DIR, folder_name,"expected_mahalanobis_mean.png"))

    ## plot expected_overlap mean
    fig, axs = plt.subplots()
    expected_overlap_mean = metrics_data['expected_overlap'].mean(dim='simulation')
    im = axs.imshow(expected_overlap_mean, cmap='viridis', origin='lower')
    axs.set_xticks(range(expected_overlap_mean.shape[1]))
    axs.set_yticks(range(expected_overlap_mean.shape[0]))
    axs.set_xticklabels(range(1, expected_overlap_mean.shape[1]+1))
    axs.set_yticklabels(range(1, expected_overlap_mean.shape[0]+1))
    axs.set_xlabel('Component')
    axs.set_ylabel('Component')
    axs.invert_yaxis()
    cbar = fig.colorbar(im)
    cbar.set_label('Expected Overlap')
    plt.savefig(os.path.join(DATA_DIR, folder_name,"expected_overlap_mean.png"))



    fig, axs = plt.subplots()
    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, iter))
    cluster_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Plot the trajectory of params["m"] in 2D space with color gradient
    for k in range(K):
        for i in range(1, iter):
            axs.plot(params["m"][i-1:i+1, k, 0], params["m"][i-1:i+1, k, 1], color=colors[i], alpha=0.5)
        axs.scatter(params["m"][-1, k, 0], params["m"][-1, k, 1], marker='o', s=10, label=f"Cluster {k+1}", c=cluster_colors[k])

    axs.set_xlim(-10, 10)
    axs.set_ylim(-10, 10)
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    axs.legend()
    plt.savefig(os.path.join(DATA_DIR, folder_name,"trajectory.png"))
    plt.close(fig)

    fig, axs = plt.subplots()
    for k in range(K):
        axs.plot(params["m"][:, k, 0], params["m"][:, k, 1], label=f"Cluster {k+1}")
    axs.set_xlim(-10, 10)
    axs.set_ylim(-10, 10)
    plt.savefig(os.path.join(DATA_DIR, folder_name,"trajectory2.png"))
    # plt.show()


    # # Plot the distances
    fig, axs = plt.subplots()
    distances = np.sqrt(np.sum((params["m"] - np.mean(params["m"], axis=1, keepdims=True))**2, axis=2))
    for k in range(K):
        axs.plot(params["alpha"][:, k] / np.sum(params["alpha"], axis=1), label=f"Cluster {k+1}")
    axs.set_xlabel("Iteration")
    axs.set_ylabel("Distance from Center")
    axs.legend()
    plt.savefig(os.path.join(DATA_DIR, folder_name,"mixtures_ratio.png"))
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
    fig, axs = plt.subplots()
    def update(i):
        axs.clear()
        artists = []

        fake_z = np.argmax(Z[i], axis=1)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for z in range(K):
            X_with_z = X[i][fake_z == z]
            scatter = axs.scatter(X_with_z[:, 0], X_with_z[:, 1], c=colors[z], s=5, alpha=0.5)
        artists.append(scatter)

        for k in range(4):
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
        axs.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc='upper right')
        # axs.set_title(f"iteration {i}")

        # plt.tight_layout()

        return artists

    ani = animation.FuncAnimation(fig, update, frames=iter, interval=500, blit=True)
    ani.save(DATA_DIR+folder_name+"/animation_colored_with_z.gif", writer="pillow")
    fig, axs = plt.subplots()
    def update(i):
        axs.clear()
        artists = []

        fake_z = np.argmax(C[i], axis=1)
        print(fake_z.shape)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for z in range(K):
            X_with_z = X[i][fake_z == z]
            scatter = axs.scatter(X_with_z[:, 0], X_with_z[:, 1], c=colors[z], s=5, alpha=0.5)
        artists.append(scatter)

        for k in range(4):
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
        axs.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc='upper right')

        # axs.set_title(f"iteration {i}")

        # plt.tight_layout()

        return artists

    ani = animation.FuncAnimation(fig, update, frames=iter, interval=500, blit=True)
    ani.save(DATA_DIR+folder_name+"/animation_colored_with_C.gif", writer="pillow")
if input('open directory? y/n') == 'y':
    os.system(f'open {DATA_DIR+folder_name}')