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
    config = json.load(open(DATA_DIR+regerence_folder_name+"/config.json"))

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
#plot variance of m
fig, axs = plt.subplots(1)
m_list = [params["m"][-1] for params in params_list]
m_array = np.array(m_list)
std_m =  np.sqrt(np.var(m_array, axis=0).mean(axis=1)) 

# Plot the variance of m
K = m_array.shape[1]
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
print(cluster_colors)
axs.bar(range(1, 1+len(std_m)), std_m, color=cluster_colors, alpha = 0.8)
axs.set_title('Standard Deviation of Mean')
axs.set_xlabel('Dimension')
axs.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "variance_m.png"))
plt.close(fig)

# 
# Calculate mean for even and odd indices
std_m_mean_even = np.mean(std_m[::2])  # Even indices (0,2,4,...)
std_m_mean_odd = np.mean(std_m[1::2])  # Odd indices (1,3,5,...)
# Plot mean variance for even and odd clusters
fig, axs = plt.subplots(1)
means = [std_m_mean_even, std_m_mean_odd]
labels = ['Sound Symbolic Words', 'Non-Sound Symbolic Words']
colors = [cluster_colors[len(cluster_colors)//2], cluster_colors[len(cluster_colors)//2+1]]  # Use first colors from even/odd clusters

axs.bar(range(1, 3), means, color=colors, alpha=0.8)
axs.set_title('Mean Standard Deviation by Cluster Type')
axs.set_xticks(range(1, 3))
axs.set_xticklabels(labels)
axs.set_ylabel('Mean Standard Deviation')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_variance_even_odd.png"))
plt.close(fig)
