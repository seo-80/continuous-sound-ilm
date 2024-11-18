import numpy as np
import xarray as xr
import sys
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import multivariate_normal
import logging
import pandas as pd
import sqlite3
from datetime import datetime
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agents import BayesianGaussianMixtureModel, BayesianGaussianMixtureModelWithContext


SAVE_RESULT = True
DATA_DIR = os.path.dirname(__file__) +"/../data/"
SAVE_LEARNING_PROCESS = True
# 合成データを生成
np.random.seed(0)
K = 8  # 混合成分数
true_K = 8
D = 2  # 次元数
N = 1000  # サンプル数
# fit_filter_name = "high_entropy"    
# fit_filter_name = "low_max_prob"
# fit_filter_name = "missunderstand"
fit_filter_name = "none"    
generate_filter_name = "none"
generate_filter_name = "high_entropy"
# generate_filter_name = "low_max_prob"
generate_filter_name = "missunderstand"

# 真の混合ガウス分布のパラメータ
true_alpha = np.array([1/true_K for _ in range(true_K)])
true_means = np.array([[5*np.cos(2*np.pi*i/true_K), 5*np.sin(2*np.pi*i/true_K)] for i in range(true_K)])
true_means = np.array([[0, 0] for _ in range(true_K)])

true_covars = np.array([[[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],])
true_covars = np.array([np.eye(D)*0.2 for _ in range(true_K)])







if fit_filter_name == "high_entropy":
    fit_filter_args = {
        "threshold": 1/16
    }
if fit_filter_name == "low_max_prob":
    fit_filter_args = {
        "threshold": 1-1/16
    }
if fit_filter_name == "missunderstand":
    fit_filter_args = {}
if fit_filter_name == "none":
    fit_filter_args = {}

if generate_filter_name == "high_entropy":
    generate_filter_args = {
        "threshold": 1/16
    }
if generate_filter_name == "low_max_prob":
    generate_filter_args = {
        "threshold": 1-1/16
    }
if generate_filter_name == "missunderstand":
    generate_filter_args = {}
if generate_filter_name == "none":
    generate_filter_args = {}

# サンプルを生成
X_0 = np.zeros((N, D))
C_0 = np.random.dirichlet(true_alpha, size=N)
z_0 = np.array([np.random.multinomial(1, c) for c in C_0])
for k in range(true_K):
    X_0[z_0[:, k] == 1] = np.random.multivariate_normal(true_means[k], true_covars[k], size=np.sum(z_0[:, k] == 1))

# モデルを初期化
alpha0 = 1.0
alpha0 = 100.0
c_alpha = np.array(
    [[2, 2, 1/16, 1/16],
    [1/16, 1/16, 2, 2]]
)
c_alpha = np.array(
    [1/K for _ in range(K)]
)
    # if c_alpha is None, agent inferes alpha and use it to generate data
# c_dirichlet_weight = [1, 1, 1, 1]
# c_alpha = np.array(
#     [1/K if i%2==0 else 1/K/4  for i in range(K)]
# )
beta0 = 1.0
nu0 = D + 2.0
m0 = np.zeros(D)
m0 = np.array([[5*np.cos(2*np.pi*i/K), 5*np.sin(2*np.pi*i/K)] for i in range(K)])
m0 = np.array([[0, 0] for i in range(K)])
W0 = np.eye(D)*0.02

#　音象徴性を変える設定
sound_symbolism_index = [2*i for i in range(K//2)]
m0 = np.array([[5*np.cos(2*np.pi*i/K), 5*np.sin(2*np.pi*i/K)] for i in range(K)])

beta0 = np.array([0.1 for _ in range(K)])
beta0[sound_symbolism_index] = 2.0


iter = 1000
agent = "BayesianGaussianMixtureModelWithContext"
# agent = "BayesianGaussianMixtureModel"
config = {
    "K": K,
    "D": D,
    "N": N,
    "c_alpha": c_alpha.tolist(),
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
    "fit_filter_func": fit_filter_name,
    "fit_filter_args": fit_filter_args,
    "generate_filter_func": generate_filter_name,
    "generate_filter_args": generate_filter_args,
}




# モデルをフィッティング
if agent == "BayesianGaussianMixtureModelWithContext":
    parent_agent = BayesianGaussianMixtureModelWithContext(K, D, alpha0, beta0, nu0, m0, W0, c_alpha, fit_filter="none", fit_filter_args=fit_filter_args, generate_filter='none', generate_filter_args=generate_filter_args)
    parent_agent.fit(xr.Dataset({
        "X": (["n", "d"], X_0),
        "C": (["n", "k"], C_0),
        "Z": (["n", "k"], z_0)
    },
    coords = {"n": np.arange(N), "d": np.arange(D), "k": np.arange(K)}
    ), max_iter=1000, tol=1e-6, random_state=0, disp_message=True)
elif agent == "BayesianGaussianMixtureModel":
    parent_agent = BayesianGaussianMixtureModel(K, D, alpha0, beta0, nu0, m0, W0, c_alpha)
    parent_agent.fit(xr.Dataset({
        "X": (["n", "d"], X_0),
    },coords={"n": np.arange(N), "d": np.arange(D)}
    ), max_iter=1000, tol=1e-6, random_state=0, disp_message=True)
params = {
    "alpha": np.zeros((iter, )+parent_agent.alpha.shape),
    "beta": np.zeros((iter, )+parent_agent.beta.shape),
    "nu": np.zeros((iter, )+parent_agent.nu.shape),
    "m": np.zeros((iter, )+parent_agent.m.shape),
    "W": np.zeros((iter, )+parent_agent.W.shape),
}
retry_counts = []
X = []
C = []
Z = []

C.append((C_0))
X.append((X_0))
Z.append((z_0))
for i in tqdm.tqdm(range(iter)):
    if agent == "BayesianGaussianMixtureModelWithContext":
        child_agent = BayesianGaussianMixtureModelWithContext(K, D, alpha0, beta0, nu0, m0, W0, c_alpha, fit_filter=fit_filter_name, fit_filter_args=fit_filter_args, generate_filter=generate_filter_name, generate_filter_args=generate_filter_args)
    elif agent == "BayesianGaussianMixtureModel":
        child_agent = BayesianGaussianMixtureModel(K, D, alpha0, beta0, nu0, m0, W0, c_alpha)
    retry_count = []
    child_agent.fit_from_agent(parent_agent, N=N)
    retry_counts.append(retry_count)
    X.append(child_agent.X)
    if agent == "BayesianGaussianMixtureModelWithContext":
        C.append(child_agent.C)
    Z.append(child_agent.Z)
    # Z.append(child_agent.Z)
    params["alpha"][i] = child_agent.alpha
    params["beta"][i] = child_agent.beta
    params["nu"][i] = child_agent.nu
    params["m"][i] = child_agent.m
    params["W"][i] = child_agent.W
    parent_agent = child_agent


#save datax
if not SAVE_RESULT:
    exit()
folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
DATA_DIR = os.path.join(DATA_DIR, folder_name)
os.makedirs(DATA_DIR, exist_ok=True)
np.save(os.path.join(DATA_DIR, "data.npy"), X)
if agent == "BayesianGaussianMixtureModelWithContext":
    np.save(os.path.join(DATA_DIR, "context.npy"), C)
    np.save(os.path.join(DATA_DIR, "Z.npy"), Z)
np.save(os.path.join(DATA_DIR, "retry_counts.npy"), retry_counts)
np.save(os.path.join(DATA_DIR, "params.npy"), params)


# Convert numpy arrays in config to lists for JSON serialization
for key, value in config.items():
    if isinstance(value, np.ndarray):
        config[key] = value.tolist()

with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
    json.dump(config, f)

