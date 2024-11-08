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
from procece_data import procece_data


parser = argparse.ArgumentParser(description='Process some data.')

parser.add_argument('folder_name',nargs="?" , type=str, default="all", help='input file path')
parser.add_argument('--remake_all', action='store_true')

folder_name = parser.parse_args().folder_name
remake_all = parser.parse_args().remake_all


#load data

DATA_DIR = os.path.dirname(__file__) +"/../data/"

matched_data = []   
if folder_name == 'all':
    folder_names = os.listdir(DATA_DIR)
    folder_names = sorted(os.listdir(DATA_DIR), reverse=True)
else:
    folder_names = [folder_name]
    remake_all = True
for folder_name in folder_names:
    if os.path.exists(DATA_DIR+folder_name+"/config.json"):
        if not os.path.exists(DATA_DIR+folder_name+"/metrics.nc"):
            print(f"procece {folder_name}")
            procece_data(folder_name)
        if os.path.exists(DATA_DIR+folder_name+"/animation.gif") and not remake_all:
            print(f"skip {folder_name}")
            continue
        print(f"load {folder_name}")
        with open(DATA_DIR+folder_name+"/config.json") as f:
            temp_config = json.load(f)
        X = np.load(DATA_DIR+folder_name+"/data.npy")
        Z = np.load(DATA_DIR+folder_name+"/Z.npy")
        C = np.load(DATA_DIR+folder_name+"/context.npy")
        K = temp_config["K"]
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
                contour = axs.contour(x, y, z, alpha=0.5, levels=[level], colors=colors[k])
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
                contour = axs.contour(x, y, z, alpha=0.5, levels=[level], colors=colors[k])
                artists.append(contour)
            axs.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc='upper right')

            # axs.set_title(f"iteration {i}")

            # plt.tight_layout()

            return artists

        ani = animation.FuncAnimation(fig, update, frames=iter, interval=500, blit=True)
        ani.save(DATA_DIR+folder_name+"/animation_colored_with_C.gif", writer="pillow")