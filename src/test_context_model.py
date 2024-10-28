import numpy as np
from agents import BayesianGaussianMixtureModelWithContext
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import multivariate_normal
import logging
import pandas as pd
import sqlite3
from datetime import datetime
import os
import json

DATA_DIR = os.path.dirname(__file__) +"/../data/"
# 合成データを生成
np.random.seed(0)
K = 4  # 混合成分数
true_K = 4
D = 2  # 次元数
N = 1000  # サンプル数
filter_name = "high_entropy"    
# filter_name = "none"    

# 真の混合ガウス分布のパラメータ
true_alpha = np.array([1/true_K, 1/true_K, 1/true_K, 1/true_K])
true_means = np.array([[5, 5], [-5, 5], [5, -5], [-5, -5]])
true_covars = np.array([[[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 1]],])
# true_covars = np.array([np.eye(D)*50 for _ in range(true_K)])



def filter_high_entropy(data, model, args):
    threshold = args["threshold"]
    p = model.predict_proba(data)
    entropy = -np.sum(p * np.log(p), axis=1)
    # print(p, entropy)
    return entropy < threshold
if filter_name == "high_entropy":
    filter_func = filter_high_entropy
if filter_name == "none":
    filter_func = lambda x, y, z: [True]
# サンプルを生成
X_0 = np.zeros((N, D))
C = np.random.dirichlet(true_alpha, size=N)
z = np.array([np.random.multinomial(1, c) for c in C])
for k in range(true_K):
    X_0[z[:, k] == 1] = np.random.multivariate_normal(true_means[k], true_covars[k], size=np.sum(z[:, k] == 1))

# モデルを初期化
alpha0 = 1.0
alpha0 = 100.0
beta0 = 1.0
nu0 = D + 2.0
m0 = np.zeros(D)
m0 = np.array([[5, 5], [-5, 5], [5, -5], [-5, -5]])
# m0 = np.array([[0, 0], [-5, 5], [5, -5], [-5, -5]])
W0 = np.eye(D)*0.02

iter = 100
config = {
    "K": K,
    "D": D,
    "N": N,
    "agent": "BayesianGaussianMixtureModelWithContext",
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

# agent = BayesianGaussianMixtureModelWithContext(K, D, alpha0, beta0, nu0, m0, W0)
# agent.fit(X_0, C, max_iter=1000, tol=1e-6, random_state=0, disp_message=True)
# # Plot X_0
# fig, ax = plt.subplots()
# ax.scatter(X_0[:, 0], X_0[:, 1])
# for k in range(K):
#     mean = agent.m[k]
#     matrix = agent.beta[k] * agent.W[k, :, :]
#     covar = np.linalg.inv(matrix)
#     x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
#     xy = np.column_stack([x.flat, y.flat])
#     z = multivariate_normal.pdf(xy, mean=mean, cov=covar).reshape(x.shape)

#     rv = multivariate_normal(mean, covar)
#     level = rv.pdf(mean) * np.exp(-0.5 * (np.sqrt(2)) ** 2)
#     ax.contour(x, y, z, alpha=0.5, levels=[level])
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# plt.show()

モデルをフィッティング
X = []
X.append((X_0))
parent_agent = BayesianGaussianMixtureModel(K, D, alpha0, beta0, nu0, m0, W0)
parent_agent.fit(X_0, max_iter=1000, tol=1e-6, random_state=0, disp_message=True)
params = {
    "alpha": np.zeros((iter, )+parent_agent.alpha.shape),
    "beta": np.zeros((iter, )+parent_agent.beta.shape),
    "nu": np.zeros((iter, )+parent_agent.nu.shape),
    "m": np.zeros((iter, )+parent_agent.m.shape),
    "W": np.zeros((iter, )+parent_agent.W.shape),
}
retry_counts = []
for i in tqdm.tqdm(range(iter)):
    child_agent = BayesianGaussianMixtureModel(K, D, alpha0, beta0, nu0, m0, W0)
    retry_count = []
    for di in range(N):
        data_stock = parent_agent.generate(N)
        count = 0
        retry = 0
        while True:
            if count >= N:
                data_stock = parent_agent.generate(N)
                count = 0
            data = data_stock[count].reshape(1, -1)
            if filter_func(data, child_agent, config['filter_args'])[0]:
                child_agent.fit(data, max_iter=1000, tol=1e-6, random_state=0, disp_message=False)
                break
            retry += 1
            count += 1
        retry_count.append(retry)
    print(retry_count)
    retry_counts.append(retry_count)
    X.append(child_agent.X)
    params["alpha"][i] = child_agent.alpha
    params["beta"][i] = child_agent.beta
    params["nu"][i] = child_agent.nu
    params["m"][i] = child_agent.m
    params["W"][i] = child_agent.W
    parent_agent = child_agent

#save data
folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
DATA_DIR = os.path.join(DATA_DIR, folder_name)
os.makedirs(DATA_DIR, exist_ok=True)
np.save(os.path.join(DATA_DIR, "data.npy"), X)
np.save(os.path.join(DATA_DIR, "retry_counts.npy"), retry_counts)
np.save(os.path.join(DATA_DIR, "params.npy"), params)

with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
    json.dump(config, f)

