import matplotlib

matplotlib.use("Agg")
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import metrics

parser = argparse.ArgumentParser(description="Process some data.")

parser.add_argument("folder_name", type=str, default=None, help="input file path")


folder_name = parser.parse_args().folder_name


# load data

DATA_DIR = os.path.dirname(__file__) + "/../data/"

X = np.load(DATA_DIR + folder_name + "/data.npy")
C = np.load(DATA_DIR + folder_name + "/context.npy")
Z = np.load(DATA_DIR + folder_name + "/Z.npy")
params = np.load(DATA_DIR + folder_name + "/params.npy", allow_pickle=True).item()
retry_counts = np.load(DATA_DIR + folder_name + "/retry_counts.npy")
iter = params["m"].shape[0]
# metrics_data = xr.open_dataset(os.path.join(DATA_DIR, folder_name, "metrics.nc"))
print("load", folder_name)
print(retry_counts)
print(X.shape)
print(C.shape)
print(Z.shape)
print(params["m"].shape)
