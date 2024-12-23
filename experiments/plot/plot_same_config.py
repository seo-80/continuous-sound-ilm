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
DATA_DIR = os.path.dirname(__file__) +"/../../data/"
OUTPUT_DIR = os.path.dirname(__file__) +"/../../figure/"

parser = argparse.ArgumentParser(description='Process some data.')

parser.add_argument('folder_name',nargs="?" , type=str, default="None", help='input file path')
regerence_folder_name = parser.parse_args().folder_name

folder_names = os.listdir(DATA_DIR)

if regerence_folder_name is not None:
    if os.path.exists(DATA_DIR+regerence_folder_name+"/config.json"):
        config = json.load(open(DATA_DIR+regerence_folder_name+"/config.json"))
    elif os.path.exists(OUTPUT_DIR+regerence_folder_name+"/config.json"):
        config = json.load(open(OUTPUT_DIR+regerence_folder_name+"/config.json"))

existing_folder_names = os.listdir(OUTPUT_DIR)
create_new_folder = True
for folder_name in existing_folder_names:
    if not os.path.exists(OUTPUT_DIR+folder_name+"/config.json"):
        continue
    temp_config = json.load(open(OUTPUT_DIR+folder_name+"/config.json"))
    if temp_config != config:
        continue
    OUTPUT_DIR = OUTPUT_DIR+folder_name+"/" 
    create_new_folder = False
    break
if create_new_folder:
    count = 0
    while True:
        if str(count) not in existing_folder_names:
            os.makedirs(OUTPUT_DIR+str(count))
            OUTPUT_DIR = OUTPUT_DIR+str(count)+"/"
            break
        count += 1

    # Save config file
    with open(OUTPUT_DIR+"config.json", "w") as f:
        json.dump(config, f)


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


params_list = []
data_list = []

for folder_name in folder_names:
    if not os.path.exists(DATA_DIR+folder_name+"/config.json"):
        continue
    temp_config = json.load(open(DATA_DIR+folder_name+"/config.json"))
    if temp_config != config:
        continue
    params_list.append(np.load(DATA_DIR+folder_name+"/params.npy", allow_pickle=True).item())
    X = np.load(DATA_DIR+folder_name+"/data.npy")
    Z = np.load(DATA_DIR+folder_name+"/Z.npy")
    C = np.load(DATA_DIR+folder_name+"/context.npy")
    data_list.append({
        "X":X,
        "Z":Z,
        "C":C
    })
print('data_num',len(params_list))
#plot variance of m

K = params_list[0]["m"][-1].shape[0]
cluster_colors = []
for i in range(K):
    if i % 2 == 0:
        # cluster_colors.append('blue')
        cluster_colors.append(plt.cm.tab20c(i//2))
    else:
        # cluster_colors.append('orange')
        cluster_colors.append(plt.cm.tab20c(4+i//2))
# bars = axs.bar(range(1, 1+len(std_m)), std_m, color=cluster_colors)
# for bar, color in zip(bars, cluster_colors):
#     bar.set_color(color)
cluster_colors = generate_double_gradation(K)

# Plot last generation
if not os.path.exists(OUTPUT_DIR+"last_generation"):
    os.makedirs(OUTPUT_DIR+"last_generation")
for i in range(len(params_list)):
    fig, axs = plt.subplots(1)
    m = params_list[i]["m"][-1]
    # Plot contours for each cluster
    for j in range(K):
        mean = m[j]
        matrix = params_list[i]["beta"][-1, j] * params_list[i]["W"][-1, j, :, :]
        covar = np.linalg.inv(matrix)
        x, y = np.meshgrid(np.linspace(-30, 30, 100), np.linspace(-30, 30, 100))
        xy = np.column_stack([x.flat, y.flat])
        z = multivariate_normal.pdf(xy, mean=mean, cov=covar).reshape(x.shape)

        rv = multivariate_normal(mean, covar)
        level = rv.pdf(mean) * np.exp(-0.5 * (1) ** 2)
        
        contour = axs.contour(x, y, z, levels=[level], colors=[cluster_colors[j]])
        contourf = axs.contourf(x, y, z, levels=[level,1], colors=[cluster_colors[j]], alpha=0.2)

        # Draw arrow from prior mean to learned mean
        prior_mean = config["m0"][j]
        axs.scatter(prior_mean[0], prior_mean[1], marker='x', color=cluster_colors[j])
        arrow = axs.arrow(prior_mean[0], prior_mean[1],
                         mean[0] - prior_mean[0],
                         mean[1] - prior_mean[1],
                         head_width=0.8, head_length=0.8,
                         fc=cluster_colors[j], ec=cluster_colors[j])
    axs.set_xlim(-30, 30)
    axs.set_ylim(-30, 30)
    plt.gca().set_aspect('equal')

    for j in range(K):
        X_last = data_list[i]["X"][-1]
        Z_last = data_list[i]["Z"][-1]
        axs.scatter(X_last[Z_last[:,j]==1, 0], X_last[Z_last[:,j]==1, 1], color=cluster_colors[j],s=2, alpha=0.2, label=f'Cluster {j+1}')
        axs.scatter(m[j, 0], m[j, 1], color=cluster_colors[j], marker='x', alpha=1, label=f'Cluster {j+1}')
    plt.savefig(os.path.join(OUTPUT_DIR, "last_generation", f"last_generation_{i}.png"), bbox_inches='tight')
    plt.close(fig)


fig, axs = plt.subplots(1)
m_list = [params["m"][-1] for params in params_list]
m_array = np.array(m_list)
# std_m =  np.sqrt(np.var(m_array, axis=0).mean(axis=1)) 
differece = m_array - np.mean(m_array, axis=0)
std_m = np.sqrt(np.mean(differece**2, axis=0)).mean(axis=1)

# Plot the variance of m
axs.bar(range(1, 1+len(std_m)), std_m, color=cluster_colors, alpha = 0.8)
# axs.set_title('Standard Deviation of Mean')
# axs.set_xlabel('Dimension')
# axs.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "standard_deciation_m.png"))
plt.close(fig)

# plot distance between m0 and mean of m
fig, axs = plt.subplots(1)
m_list = [params["m"][-1] for params in params_list]
m_array = np.array(m_list)
mean_m = np.mean(m_array, axis=0)
# Calculate distance between mean_m and initial means (m0)
distance = np.linalg.norm(mean_m - np.array(config["m0"]), axis=1)
labels = ['Sound Symbolic Words', 'Non-Sound Symbolic Words']
distance_even = distance[::2]
distance_odd = distance[1::2]
# Plot distances as bar chart
colors = [cluster_colors[len(cluster_colors)//2], cluster_colors[len(cluster_colors)//2+1]]  # Use first colors from even/odd clusters
axs.bar(range(1, 3), [np.mean(distance_even), np.mean(distance_odd)], color=colors, alpha=0.8)
# axs.set_title('Distance between Initial and Mean Final Position')
# axs.set_xlabel('Cluster')
# axs.set_ylabel('Distance')


plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distance_m0_mean_m.png"))
plt.close(fig)

# 
# Calculate mean for even and odd indices
std_m_mean_even = np.mean(std_m[::2])  # Even indices (0,2,4,...)
std_m_mean_odd = np.mean(std_m[1::2])  # Odd indices (1,3,5,...)
# Plot mean SD for even and odd clusters
fig, axs = plt.subplots(1)
means = [std_m_mean_even, std_m_mean_odd]
labels = ['Sound Symbolic Words', 'Non-Sound Symbolic Words']
colors = [cluster_colors[len(cluster_colors)//2], cluster_colors[len(cluster_colors)//2+1]]  # Use first colors from even/odd clusters

axs.bar(range(1, 3), means, color=colors, alpha=0.8)
# axs.set_title('Mean Standard Deviation by Cluster Type')
axs.set_xticks(range(1, 3))
axs.set_xticklabels(labels)
# axs.set_ylabel('Mean Standard Deviation')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_standard_deciation_even_odd.png"))
plt.close(fig)

# plrt mean variance for even and odd clusters
fig, axs = plt.subplots(1)
var_m = np.mean(differece**2, axis=0)
var_m_mean_even = np.mean(var_m[::2])  # Even indices (0,2,4,...)
var_m_mean_odd = np.mean(var_m[1::2])  # Odd indices (1,3,5,...)
vars = [var_m_mean_even, var_m_mean_odd]
labels = ['Sound Symbolic Words', 'Non-Sound Symbolic Words']   
colors = [cluster_colors[len(cluster_colors)//2], cluster_colors[len(cluster_colors)//2+1]]  # Use first colors from even/odd clusters
axs.bar(range(1, 3), vars, color=colors, alpha=0.8)
axs.set_xticks(range(1, 3))
axs.set_xticklabels(labels)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_variance_even_odd.png"))


# plot final generation mean
fig, axs = plt.subplots()
m_list = [params["m"][-1] for params in params_list]
m_array = np.array(m_list)
K = m_array.shape[1]    
cluster_colors = generate_double_gradation(K)
axs.set_xlim(-30, 30)
axs.set_ylim(-30, 30)
plt.gca().set_aspect('equal')
for i in range(K):
    axs.scatter(config["m0"][i][0], config["m0"][i][1], color=cluster_colors[i], marker='x',  alpha=1, label=f'Initial {i+1}')
    axs.scatter(m_array[:, i, 0], m_array[:, i, 1], color=cluster_colors[i],s=5, alpha=0.8, label=f'Cluster {i+1}')
# axs.set_title('Final Generation Mean')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_generation_mean.png"))
plt.close(fig)

# plot mean of final generation mean 
fig, axs = plt.subplots()
mean_m = np.mean(m_array, axis=0)
axs.set_xlim(-30, 30)
axs.set_ylim(-30, 30)
plt.gca().set_aspect('equal')
for i in range(K):
    axs.scatter(config["m0"][i][0], config["m0"][i][1], color=cluster_colors[i], marker='x',  alpha=1, label=f'Initial {i+1}')
    axs.scatter(mean_m[i, 0], mean_m[i, 1], color=cluster_colors[i],s=5, alpha=0.8, label=f'Cluster {i+1}')
# Draw arrows from initial means (m0) to mean final means
for i in range(K):
    axs.arrow(config["m0"][i][0], config["m0"][i][1],
              mean_m[i,0] - config["m0"][i][0],
              mean_m[i,1] - config["m0"][i][1],
              head_width=0.8, head_length=0.8,
              fc=cluster_colors[i], ec=cluster_colors[i],
              alpha=0.8)
# axs.set_title('Mean of Final Generation Mean')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_final_generation_mean.png"))
plt.close(fig)



fig, axs = plt.subplots()
axs.set_xlim(-30, 30)
axs.set_ylim(-30, 30)
plt.gca().set_aspect('equal')
for i in range(K):
    axs.scatter(config["m0"][i][0], config["m0"][i][1], color=cluster_colors[i], marker='x',  alpha=1, label=f'Initial {i+1}')
    axs.scatter(m_array[:, i, 0], m_array[:, i, 1], color=cluster_colors[i],s=5, alpha=0.8, label=f'Cluster {i+1}')
# Draw lines from initial means (m0) to final means for all clusters
for i in range(K):
    for j in range(len(m_array)):
        axs.plot([config["m0"][i][0], m_array[j,i,0]], 
                 [config["m0"][i][1], m_array[j,i,1]], 
                 color=cluster_colors[i], alpha=0.4, linewidth=0.5)
# axs.set_title('Final Generation Mean')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_generation_mean_with_arrow.png"))
plt.close(fig)

# histogram of distance between m0 and m
fig, axs = plt.subplots()
m_list = [params["m"][-1] for params in params_list]
m_array = np.array(m_list)
K = m_array.shape[1]
cluster_colors = generate_double_gradation(K)
distance = np.linalg.norm(m_array - np.array(config["m0"]), axis=2)
distance = np.concatenate([distance[:,::2], distance[:,1::2]], axis=1)

# Calculate histograms manually for both clusters
bins = 20
range_min = np.min(distance)
range_max = np.max(distance)
bin_edges = np.linspace(range_min, range_max, bins + 1)
bin_width = (range_max - range_min) / bins
bar_width = bin_width * 0.35  # Make bars narrower to fit side by side
labels = ['Sound Symbolic Words', 'Non-Sound Symbolic Words']
for i in range(2):
    hist, _ = np.histogram(distance[:, i], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Offset the bars for each cluster
    offset = (i - 0.5) * bar_width
    axs.bar(bin_centers + offset, hist, width=bar_width, 
            color=cluster_colors[len(cluster_colors)//2+i], alpha=1.0, 
            label=labels[i])

# axs.set_title('Distance between Initial and Final Generation Mean (Even/Odd)')
# axs.set_xlabel('Distance')
# axs.set_ylabel('Frequency')
axs.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distance_m0_m.png"))
plt.close(fig)

# Plot histogram of distance between (0,0) and m for all clusters
fig, axs = plt.subplots()
distance = np.linalg.norm(m_array - np.array(config["m0"]), axis=2)
for i in range(len(m_array)):
    # Calculate distance from origin for each mean
    origin_dist = np.linalg.norm(m_array[i], axis=1)
    # Invert distance values for means inside radius 10 circle
    inside_circle = origin_dist <= 10
    distance[i, inside_circle] = -distance[i, inside_circle]
distance = np.concatenate([distance[:,::2], distance[:,1::2]], axis=1)
distance = distance - np.linalg.norm(np.array(config["m0"]), axis=1)
bins = 20
range_min = np.min(distance)
range_max = np.max(distance)
bin_edges = np.linspace(range_min, range_max, bins + 1)
bin_width = (range_max - range_min) / bins
bar_width = bin_width * 0.35  # Make bars narrower to fit side by side
labels = ['Sound Symbolic Words', 'Non-Sound Symbolic Words']
for i in range(2):
    hist, _ = np.histogram(distance[:, i], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Offset the bars for each cluster
    offset = (i - 0.5) * bar_width
    axs.bar(bin_centers + offset, hist, width=bar_width, 
            color=cluster_colors[len(cluster_colors)//2+i], alpha=1.0, 
            label=labels[i])

# axs.set_title('Distance between (0,0) and Final Generation Mean (Even/Odd)')
# axs.set_xlabel('Distance')
# axs.set_ylabel('Frequency')
axs.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distance_m_inv.png"))
plt.close(fig)
# 上の二つの図を合わせたもの
fig, axs = plt.subplots()
axs.set_xlim(-30, 30)
axs.set_ylim(-30, 30)
plt.gca().set_aspect('equal')
mean_m = np.mean(m_array, axis=0)
# cov_m = np.array([np.cov(m_array[i, :, 0], m_array[i, :, 1]) for i in range(K)])
differece = m_array - np.mean(m_array, axis=0)
var_m = np.mean(differece**2, axis=0).mean(axis=1)
print(var_m.shape)
cov_m = np.array([np.eye(2) * var_m[i] for i in range(K)])



for i in range(K):
    axs.scatter(config["m0"][i][0], config["m0"][i][1], color=cluster_colors[i], marker='x',  alpha=1, label=f'Initial {i+1}')
    axs.scatter(m_array[:, i, 0], m_array[:, i, 1], color=cluster_colors[i],s=5, alpha=0.2, label=f'Cluster {i+1}')
    # for j in range(len(m_array)):
    #     axs.plot([config["m0"][i][0], m_array[j,i,0]], 
    #              [config["m0"][i][1], m_array[j,i,1]], 
    #              color=cluster_colors[i], alpha=0.6, linewidth=0.2)
# Draw circles for 1 standard deviation
for i in range(K):
    x, y = np.meshgrid(np.linspace(-30, 30, 200), np.linspace(-30, 30, 200))
    xy = np.column_stack([x.flat, y.flat])
    z = multivariate_normal.pdf(xy, mean=mean_m[i], cov=cov_m[i]).reshape(x.shape)

    rv = multivariate_normal(mean_m[i], cov_m[i])
    level = rv.pdf(mean_m[i]) * np.exp(-0.5 * 1 ** 2)
    levels = [level]
    contour = axs.contour(x, y, z, levels=levels, colors=[cluster_colors[i]])
    contourf = axs.contourf(x, y, z, levels=[level,1], colors=[cluster_colors[i]], alpha=0.2)
for i in range(K):
    axs.arrow(config["m0"][i][0], config["m0"][i][1],
              mean_m[i,0] - config["m0"][i][0],
              mean_m[i,1] - config["m0"][i][1],
              head_width=0.8, head_length=0.8,
              fc=cluster_colors[i], ec=cluster_colors[i],
              alpha=0.8)
# axs.set_title('Final Generation Mean')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_generation_mean_with_arrow_and_distance_with_count.png"))
