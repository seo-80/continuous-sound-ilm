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

if regerence_folder_name is not "None":
    config = json.load(open(DATA_DIR+regerence_folder_name+"/config.json"))
values_key = "beta0"
base_values = [0.1, 1, 10, 100]
base_values = [1,2,3,4,5,6,7,8,9,10]
values_list = [
    [i for _ in range(4)] for i in base_values
    ]

existing_folder_names = os.listdir(OUTPUT_DIR)
create_new_folder = True

def compute_convergence_time(history_m, true_m, threshold=0.01):

    history_m = history_m['m'].values
    for i in range(len(history_m)):
        if np.linalg.norm(history_m[i] - true_m) < threshold:
            return i
    print("Error: Convergence time is not found.")
    return len(history_m)

for folder_name in existing_folder_names:
    if not os.path.exists(OUTPUT_DIR+folder_name+"/config.json"):
        continue
    temp_config = json.load(open(OUTPUT_DIR+folder_name+"/config.json"))
    if temp_config != config:
        continue
    if os.path.exists(OUTPUT_DIR+folder_name+"/history.nc"):
        continue

    create_new_folder = False
    OUTPUT_DIR = OUTPUT_DIR+folder_name+"/"

    break
if create_new_folder:
    count = 0
    while True:
        if str(count) not in existing_folder_names:
            os.makedirs(OUTPUT_DIR+str(count))
            OUTPUT_DIR = OUTPUT_DIR+str(count)+"/"
            break
        count += 1

params_list = []
data_list = []
def generate_gradation(n):
    colors = []
    start_color = (1.0, 0.2, 0.0)  # Red in RGB

    end_color = (0.0, 0.2, 1.0)


    for i in range(n):
        t = i / (n-1)
        color = tuple(start_color[j] + (end_color[j] - start_color[j]) * t for j in range(3))
        colors.append(color)
    
    return colors
for value in values_list:
    config[values_key] = value
    exist = False
    for folder_name in folder_names:
        if not os.path.exists(DATA_DIR+folder_name+"/config.json"):
            continue
        temp_config = json.load(open(DATA_DIR+folder_name+"/config.json"))
        if temp_config != config:
            continue
        if not os.path.exists(DATA_DIR+folder_name+"/history.nc"):
            continue
        X = np.load(DATA_DIR+folder_name+"/data.npy")
        Z = np.load(DATA_DIR+folder_name+"/Z.npy")
        C = np.load(DATA_DIR+folder_name+"/context.npy")
        params = np.load(DATA_DIR+folder_name+"/params.npy", allow_pickle=True).item()
        params_list.append(params)
        history_m = xr.open_dataset(DATA_DIR+folder_name+"/history.nc", drop_variables=list(params.keys() - {"m"}))
        data_list.append({
            "X":X,
            "Z":Z,
            "C":C,
            "params":params,
            "history_m":history_m
        })
        exist = True
        break
    if not exist:
        print(f"Error: {value} is not found.")
        sys.exit(1)

if len(params_list) != len(values_list):
    print("Error: len(params_list) != len(values_list)")
    print(len(params_list), len(values_list))
    sys.exit(1)
#plot Convergence time

convergence_time_list = []
colors = generate_gradation(len(data_list))
for data in data_list:
    history_m = data["history_m"]
    convergence_times = [compute_convergence_time(history_m.isel(iter=i), data['params']['m'][i-1],1/2) for i in range(1,history_m.dims['iter']-1)]
    convergence_time = np.mean(convergence_times)
    print(convergence_time)
    convergence_time_list.append(convergence_time)
fig, axs = plt.subplots(1)
axs.bar(range(1, len(convergence_time_list) + 1), convergence_time_list, color=colors)
# axs.set_title('Convergence Time for Different c_alpha Configurations')
axs.set_xticks(range(1, len(convergence_time_list) + 1))
axs.set_xticklabels([base_values[i] for i in range(len(convergence_time_list))])

# axs.set_ylabel('Convergence Time')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "convergence_time.png"))
plt.close(fig)



        
        

print('folder_name:', OUTPUT_DIR)