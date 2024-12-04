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
import colorsys





sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import metrics
from experiments.procece_data import procece_data


parser = argparse.ArgumentParser(description='Process some data.')

parser.add_argument('folder_name',nargs="?" , type=str, default="all", help='input file path')
parser.add_argument('--remake_all', action='store_true')
parser.add_argument('--latest', action='store_true')

folder_name = parser.parse_args().folder_name
remake_all = parser.parse_args().remake_all
latest = parser.parse_args().latest

if latest and remake_all:
    print("latest and remake_all are exclusive")
    exit()


#load data

DATA_DIR = os.path.dirname(__file__) +"/../../data/"

def generate_colors(n):
    """
    n個の視覚的に区別しやすい色を生成する
    
    Parameters:
    n (int): 必要な色の数
    
    Returns:
    list: RGBカラーコードのリスト
    """
    colors = []
    for i in range(n):
        # HSVカラースペースで均等に色相を分割
        hue = i / n
        saturation = 0.7  # 彩度
        value = 0.9      # 明度
        
        # HSVからRGBに変換し、文字列形式に変更
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return colors
def generate_double_gradation(n):
    """
    n個の視覚的に区別しやすい色を生成する
    
    Parameters:
    n (int): 必要な色の数
    
    Returns:
    list: RGBカラーコードのリスト
    """
    # Start with red and end with orange
    start_color1 = (0.0, 0.0, 1.0)  # Blue in RGB
    start_color2 = (1.0, 0.0, 0.0)  # Red in RGB
    end_color1 = (0.0, 0.8, 1.0)   # Light blue in RGB
    end_color2 = (1.0, 0.8, 0.0)   # Orange in RGB
    
    colors = []
    for i in range(n):
        if i % 2 == 0:
            t = (i // 2) / ((n-1) // 2)  # Normalized position
            color = tuple(start_color1[j] + (end_color1[j] - start_color1[j]) * t for j in range(3))
        else:
            t = ((i-1) // 2) / (n // 2)  # Normalized position
            color = tuple(start_color2[j] + (end_color2[j] - start_color2[j]) * t for j in range(3))
        colors.append(color)
    
    return colors

matched_data = []  
if folder_name == 'all':
    folder_names = os.listdir(DATA_DIR)
    folder_names = sorted(os.listdir(DATA_DIR), reverse=True)
    if latest:
        folder_names = sorted(folder_names, reverse=True)[:1]
        remake_all = True
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
        
        if os.path.exists(DATA_DIR+folder_name+"/excluded_data.nc"):
            try:
                excluded_data = xr.open_dataset(DATA_DIR+folder_name+"/excluded_data.nc")
            except:
                excluded_data = None

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
        # Determine x_lim and y_lim based on X values
        lim = np.max(np.abs(params["m"]))
        lim = np.ceil(lim / 10) * 10
        x_lim = (-lim, lim)
        y_lim = (-lim, lim)

            

        iter = params["m"].shape[0]
        # cluster_colors = generate_colors(K)
        # cluster_colors = plt.cm.tab20b(np.linspace(0, 1, K))
        # cluster_colors = []
        # for i in range(K):
        #     if i % 2 == 0:
        #         # cluster_colors.append('blue')
        #         cluster_colors.append(plt.cm.tab20c(i//2))
        #     else:
        #         # cluster_colors.append('orange')
        #         cluster_colors.append(plt.cm.tab20c(4+i//2))
        cluster_colors = generate_double_gradation(K)

        if os.path.exists(DATA_DIR+folder_name+"/history.nc"):
            history_m = xr.open_dataset(DATA_DIR+folder_name+"/history.nc", drop_variables=list(params.keys() - {"m"}))
            history_m_diff = np.array([history_m['m'][i] - history_m['m'][i-1][-1] for i in range(1, len(history_m['m']))])
            history_m_diff = np.linalg.norm(history_m_diff, axis=-1)

            history_m_diff = np.mean(history_m_diff, axis=0)
            fig, axs = plt.subplots()
            for k in range(K):
                axs.plot(history_m_diff[:,k], label=f"Cluster {k+1}")
            axs.set_xlabel("sumple number")
            axs.set_ylabel("Difference")
            axs.legend()
            plt.savefig(os.path.join(DATA_DIR, folder_name, "history_m_diff.png"))

            # plot animation of learning process of last generation
            fig, axs = plt.subplots()
            last_generation_history = xr.open_dataset(DATA_DIR+folder_name+"/history.nc").sel(iter=99)
            print(last_generation_history)
            def update(i):
                axs.clear()
                artists = []

                # Plot data points up to current time i
                fake_z = np.argmax(Z[-1], axis=1)
                for z in range(K):
                    X_with_z = X[-1][fake_z == z][:i]
                    scatter = axs.scatter(X_with_z[:, 0], X_with_z[:, 1], c=[cluster_colors[z]], s=2, alpha=0.2)
                    artists.append(scatter)

                # Plot current distribution
                for k in range(K):
                    mean = last_generation_history["m"][i][k]
                    matrix = (last_generation_history["beta"][i][k].values * 
                                last_generation_history["W"][i][k, :, :].values)
                    covar = np.linalg.inv(matrix)
                    axs.set_xlim(-30, 30)
                    axs.set_ylim(-30, 30)
                    x, y = np.meshgrid(np.linspace(*x_lim, 100), np.linspace(*y_lim, 100))
                    xy = np.column_stack([x.flat, y.flat])
                    z = multivariate_normal.pdf(xy, mean=mean, cov=covar).reshape(x.shape)

                    rv = multivariate_normal(mean, covar)
                    level = rv.pdf(mean) * np.exp(-0.5 * (np.sqrt(2)) ** 2)
                    # levels = level * np.linspace(0, 5, 10)
                    levels = [level]
                    contour = axs.contour(x, y, z, levels=levels, colors=[cluster_colors[k]])
                    artists.append(contour)
                    contourf = axs.contourf(x, y, z, levels=[level,1], colors=[cluster_colors[k]], alpha=0.2)
                    artists.append(contourf)
                
                # Plot prior means and arrows to current means
                for k in range(K):
                    prior_mean = temp_config["m0"][k]
                    marker = axs.scatter(prior_mean[0], prior_mean[1], marker='x', color=cluster_colors[k])
                    artists.append(marker)
                    learned_mean = last_generation_history["m"][i][k]
                    arrow = axs.arrow(prior_mean[0], prior_mean[1], 
                                    learned_mean[0] - prior_mean[0], 
                                    learned_mean[1] - prior_mean[1], 
                                    head_width=0.8, head_length=0.8, 
                                    fc=cluster_colors[k], ec=cluster_colors[k])
                    artists.append(arrow)

                return artists
            # Create animation
            anim = animation.FuncAnimation(fig, update, frames=len(params["m"]), interval=50, blit=True)
            
            # Save animation
            anim.save(os.path.join(DATA_DIR, folder_name, "learning_animation.gif"), writer='pillow')
            plt.close(fig)
        

        # Create a list of alternating blue and orange colors
        
        # plot mean step difference
        fig, axs = plt.subplots()
        # Calculate mean step difference for each cluster
        mean_step_diff = np.mean(np.linalg.norm(np.diff(params["m"][len(params['m'])//10:], axis=0), axis=-1), axis=0)
        # Create bar plot of mean step differences
        axs.bar(range(1, K+1), mean_step_diff)
        axs.set_xticks(range(1, K+1))
        axs.set_xticklabels([f"Cluster {k+1}" for k in range(K)])
        axs.set_ylabel("Mean Step Difference")
        axs.set_title("Mean Step Differences by Cluster")
        plt.tight_layout() 
        plt.savefig(os.path.join(DATA_DIR, folder_name, "mean_step_diff.png"))
        plt.close(fig)
        
        # plot std of mean
        fig, axs = plt.subplots()
        # Calculate variance of mean for each cluster
        std_m =  np.sqrt(np.var(params["m"][len(params['m'])*5//10:], axis=0).mean(axis=1)) 
        # Create bar plot of variance of mean
        axs.bar(range(1, K+1), std_m)
        axs.set_xticks(range(1, K+1))
        axs.set_xticklabels([f"Cluster {k+1}" for k in range(K)])
        axs.set_ylabel("Standard Deviation of Mean")
        axs.set_title("Standard Deviation of Mean by Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, folder_name, "std_m.png"))



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
        # Extend cluster_colors to support up to 32 clusters
        # Extend cluster_colors to support up to 32 clusters


        # Plot the trajectory of params["m"] in 2D space with color gradient
        for k in range(K):
            for i in range(1, iter):
                axs.plot(params["m"][i-1:i+1, k, 0], params["m"][i-1:i+1, k, 1], color=colors[i], alpha=0.5)
            axs.scatter(params["m"][-1, k, 0], params["m"][-1, k, 1], marker='o', s=10, label=f"Cluster {k+1}", c=[cluster_colors[k]])

        axs.set_xlim(x_lim)
        axs.set_ylim(y_lim)
        axs.set_xlabel("X")
        axs.set_ylabel("Y")
        axs.legend()
        plt.savefig(os.path.join(DATA_DIR, folder_name,"trajectory.png"))
        plt.close(fig)

        fig, axs = plt.subplots()
        for k in range(K):
            axs.plot(params["m"][:, k, 0], params["m"][:, k, 1], label=f"Cluster {k+1}", c=cluster_colors[k])
        axs.set_xlim(x_lim)
        axs.set_ylim(y_lim)
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

            scatter = axs.scatter(X[i][:, 0], X[i][:, 1], s=2, alpha=0.5)
            artists.append(scatter)
            if excluded_data is not None:
                if 'iter' in excluded_data.dims:
                    excluded = excluded_data['X'][i]
                    scatter = axs.scatter(excluded[:, 0], excluded[:, 1], s=2, alpha=0.5, c='red')
                    artists.append(scatter)

            for k in range(K):
                mean = params["m"][i, k]
                matrix = params["beta"][i, k] * params["W"][i, k, :, :]
                covar = np.linalg.inv(matrix)
                axs.set_xlim(x_lim)
                axs.set_ylim(y_lim)
                x, y = np.meshgrid(np.linspace(*x_lim, 100), np.linspace(*y_lim, 100))
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
            for z in range(K):
                X_with_z = X[i][fake_z == z]
                scatter = axs.scatter(X_with_z[:, 0], X_with_z[:, 1], c=[cluster_colors[z]], s=2, alpha=0.2)
            artists.append(scatter)

            for k in range(K):
                mean = params["m"][i, k]
                matrix = params["beta"][i, k] * params["W"][i, k, :, :]
                covar = np.linalg.inv(matrix)
                # axs.set_xlim(x_lim)
                # axs.set_ylim(y_lim)
                axs.set_xlim(-30, 30)
                axs.set_ylim(-30, 30)
                x, y = np.meshgrid(np.linspace(*x_lim, 100), np.linspace(*y_lim, 100))
                xy = np.column_stack([x.flat, y.flat])
                z = multivariate_normal.pdf(xy, mean=mean, cov=covar).reshape(x.shape)

                rv = multivariate_normal(mean, covar)
                level = rv.pdf(mean) * np.exp(-0.5 * (np.sqrt(2)) ** 2)
                
                contour = axs.contour(x, y, z, levels=[level], colors=[cluster_colors[k]])
                artists.append(contour)
                contourf = axs.contourf(x, y, z, levels=[level,1], colors=[cluster_colors[k]], alpha=0.2)
                artists.append(contourf)
            
            # 事前分布中心から学習後の中心までの矢印を描画
            for k in range(K):
                prior_mean = temp_config["m0"][k]
                marker = axs.scatter(prior_mean[0], prior_mean[1], marker='x', color=cluster_colors[k])
                artists.append(marker)
                learned_mean = params["m"][i, k]
                arrow = axs.arrow(prior_mean[0], prior_mean[1], 
                                  learned_mean[0] - prior_mean[0], 
                                  learned_mean[1] - prior_mean[1], 
                                  head_width=0.8, head_length=0.8, 
                                  fc=cluster_colors[k], ec=cluster_colors[k])
                artists.append(arrow)


            # axs.legend([f'Cluster {i+1}' for i in range(K)], loc='upper right')
            # axs.set_title(f"iteration {i}")

            # plt.tight_layout()

            return artists

        # ani = animation.FuncAnimation(fig, update, frames=iter, interval=500, blit=True)
        # ani.save(DATA_DIR+folder_name+"/animation_colored_with_z.gif", writer="pillow")
        # fig, axs = plt.subplots()
        # def update(i):
        #     axs.clear()
        #     artists = []

        #     fake_z = np.argmax(C[i], axis=1)
        #     for z in range(K):
        #         X_with_z = X[i][fake_z == z]
        #         scatter = axs.scatter(X_with_z[:, 0], X_with_z[:, 1], c=[cluster_colors[z]], s=2, alpha=0.5)
        #     artists.append(scatter)

        #     for k in range(K):
        #         mean = params["m"][i, k]
        #         matrix = params["beta"][i, k] * params["W"][i, k, :, :]
        #         covar = np.linalg.inv(matrix)
        #         axs.set_xlim(x_lim)
        #         axs.set_ylim(y_lim)
        #         x, y = np.meshgrid(np.linspace(*x_lim, 100), np.linspace(*y_lim, 100))
        #         xy = np.column_stack([x.flat, y.flat])
        #         z = multivariate_normal.pdf(xy, mean=mean, cov=covar).reshape(x.shape)

        #         rv = multivariate_normal(mean, covar)
        #         level = rv.pdf(mean) * np.exp(-0.5 * (np.sqrt(2)) ** 2)
        #         contour = axs.contour(x, y, z, alpha=0.5, levels=[level], colors=[cluster_colors[k]])
        #         artists.append(contour)
        #     axs.legend([f'Cluster {i+1}' for i in range(K)], loc='upper right')

            # axs.set_title(f"iteration {i}")

            # plt.tight_layout()

            return artists

        ani = animation.FuncAnimation(fig, update, frames=iter, interval=500, blit=True)
        ani.save(DATA_DIR+folder_name+"/animation_colored_with_C.gif", writer="pillow")


        
        plt.cla()