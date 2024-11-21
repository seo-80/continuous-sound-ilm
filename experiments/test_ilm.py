import numpy as np
import xarray as xr
import sys
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agents import BayesianGaussianMixtureModel, BayesianGaussianMixtureModelWithContext

@dataclass
class ExperimentConfig:
    K: int  # 混合成分数
    D: int  # 次元数
    N: int  # サンプル数
    agent: str
    true_K: int
    alpha0: float
    beta0: float
    nu0: float
    c_alpha: np.ndarray
    m0: np.ndarray
    W0: np.ndarray
    iter: int
    fit_filter_name: str
    generate_filter_name: str
    fit_filter_args: Dict[str, Any]
    generate_filter_args: Dict[str, Any]

    @classmethod
    def create_default_config(cls) -> 'ExperimentConfig':
        K = 8
        D = 2
        N = 1000
        true_K = 8
        
        # Default parameters
        c_alpha = np.array([1/K for _ in range(K)])
        alpha0 = 100.0
        beta0 = np.array([1 if i%2 ==0 else 0.1 for i in range(K)])
        nu0 = D + 2.0
        m0_range = 10
        m0 = np.array([[m0_range*np.cos(2*np.pi*i/K), m0_range*np.sin(2*np.pi*i/K)] for i in range(K)])
        W0 = np.eye(D)*0.02

        return cls(
            K=K, D=D, N=N, true_K=true_K,
            agent="BayesianGaussianMixtureModelWithContext",
            alpha0=alpha0, beta0=beta0, nu0=nu0,
            c_alpha=c_alpha, m0=m0, W0=W0,
            iter=100,
            fit_filter_name="none",
            generate_filter_name="missunderstand",
            fit_filter_args={},
            generate_filter_args={},
        )
    def load_config(self, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(
            K=config["K"],
            D=config["D"],
            N=config["N"],
            true_K=config["true_K"],
            agent=config["agent"],
            alpha0=config["alpha0"],
            beta0=config["beta0"],
            nu0=config["nu0"],
            c_alpha=np.array(config["c_alpha"]),
            m0=np.array(config["m0"]),
            W0=np.array(config["W0"]),
            iter=config["iter"],
            fit_filter_name=config["fit_filter_name"],
            generate_filter_name=config["generate_filter_name"],
            fit_filter_args=config["fit_filter_args"],
            generate_filter_args=config["generate_filter_args"],
            true_alpha=np.array(config["true_alpha"]),
            true_means=np.array(config["true_means"]),
            true_covars=np.array(config["true_covars"])
        )

class ExperimentManager:
    def __init__(self, config: ExperimentConfig, save_dir: str, track_learning: bool = False):
        self.config = config
        self.save_dir = save_dir
        self.setup_data_directory()
        self.track_learning = track_learning
        
        # Initialize storage for results
        self.params = {
            "alpha": np.zeros((config.iter,) + (config.K,)),
            "beta": np.zeros((config.iter,) + (config.K,)),
            "nu": np.zeros((config.iter,) + (config.K,)),
            "m": np.zeros((config.iter, config.K, config.D)),
            "W": np.zeros((config.iter, config.K, config.D, config.D))
        }
        self.X = []
        self.C = []
        self.Z = []
        self.excluded_data = []
        self.retry_counts = []
        if self.track_learning:
            self.history = xr.Dataset({
                "alpha": (["iter", "n", "k"], np.zeros((config.iter, config.N,  config.K))),
                "beta": (["iter", "n", "k"], np.zeros((config.iter, config.N,  config.K))),
                "nu": (["iter", "n", "k"], np.zeros((config.iter, config.N,  config.K))),
                "m": (["iter", "n", "k", "d"], np.zeros((config.iter, config.N,  config.K, config.D))),
                "W": (["iter", "n", "k", "d", "d"], np.zeros((config.iter, config.N,  config.K, config.D, config.D)))
            },
            coords={"iter": np.arange(config.iter), "n": np.arange(config.N), "K": np.arange(config.K), "d": np.arange(config.D)})



    def setup_data_directory(self):
        """実験データ保存用のディレクトリを設定"""
        folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = os.path.join(self.save_dir, folder_name)
        os.makedirs(self.save_path, exist_ok=True)

    def generate_initial_data(self) -> tuple:
        """初期データの生成"""
        np.random.seed(0)
        X_0 = np.zeros((self.config.N, self.config.D))
        C_0 = np.random.dirichlet(self.config.true_alpha, size=self.config.N)
        z_0 = np.array([np.random.multinomial(1, c) for c in C_0])
        
        for k in range(self.config.true_K):
            X_0[z_0[:, k] == 1] = np.random.multivariate_normal(
                self.config.true_means[k],
                self.config.true_covars[k],
                size=np.sum(z_0[:, k] == 1)
            )
            
        return X_0, C_0, z_0

    def create_agent(self, is_parent: bool = False) -> Any:
        """エージェントの作成"""
        if self.config.agent == "BayesianGaussianMixtureModelWithContext":
            return BayesianGaussianMixtureModelWithContext(
                self.config.K, self.config.D,
                self.config.alpha0, self.config.beta0,
                self.config.nu0, self.config.m0, self.config.W0,
                self.config.c_alpha,
                fit_filter="none" if is_parent else self.config.fit_filter_name,
                fit_filter_args=self.config.fit_filter_args,
                generate_filter="none" if is_parent else self.config.generate_filter_name,
                generate_filter_args=self.config.generate_filter_args,
                track_learning=self.track_learning
            )
        else:
            return BayesianGaussianMixtureModel(
                self.config.K, self.config.D,
                self.config.alpha0, self.config.beta0,
                self.config.nu0, self.config.m0, self.config.W0,
                self.config.c_alpha,
                track_learning=self.track_learning
            )

    def fit_parent_agent(self, X_0: np.ndarray, C_0: np.ndarray, z_0: np.ndarray) -> Any:
        """親エージェントの学習"""
        parent_agent = self.create_agent(is_parent=True)
        
        if self.config.agent == "BayesianGaussianMixtureModelWithContext":
            data = xr.Dataset({
                "X": (["n", "d"], X_0),
                "C": (["n", "k"], C_0),
                "Z": (["n", "k"], z_0)
            }, coords={"n": np.arange(self.config.N), "d": np.arange(self.config.D), "k": np.arange(self.config.K)})
        else:
            data = xr.Dataset({
                "X": (["n", "d"], X_0),
            }, coords={"n": np.arange(self.config.N), "d": np.arange(self.config.D)})
            
        parent_agent.fit(data, max_iter=1000, tol=1e-6, random_state=0, disp_message=True)
        return parent_agent

    def run_experiment(self):
        """実験の実行"""
        parent_agent = self.create_agent()

        for i in tqdm.tqdm(range(self.config.iter)):
            child_agent = self.create_agent()
            retry_count = []
            child_agent.fit_from_agent(parent_agent, N=self.config.N)
            
            self.retry_counts.append(retry_count)
            self.X.append(child_agent.X)
            if self.config.agent == "BayesianGaussianMixtureModelWithContext":
                self.C.append(child_agent.C)
            self.Z.append(child_agent.Z)
            
            # パラメータの保存
            self.params["alpha"][i] = child_agent.alpha
            self.params["beta"][i] = child_agent.beta
            self.params["nu"][i] = child_agent.nu
            self.params["m"][i] = child_agent.m
            self.params["W"][i] = child_agent.W
            self.excluded_data.append(child_agent.excluded_data)
            if self.track_learning:
                self.history['alpha'][i] = child_agent.history['alpha']
                self.history['beta'][i] = child_agent.history['beta']
                self.history['nu'][i] = child_agent.history['nu']
                self.history['m'][i] = child_agent.history['m']
                self.history['W'][i] = child_agent.history['W']

            
            parent_agent = child_agent

    def save_results(self):
        """結果の保存"""
        np.save(os.path.join(self.save_path, "data.npy"), self.X)
        if self.config.agent == "BayesianGaussianMixtureModelWithContext":
            np.save(os.path.join(self.save_path, "context.npy"), self.C)
            np.save(os.path.join(self.save_path, "Z.npy"), self.Z)
        np.save(os.path.join(self.save_path, "retry_counts.npy"), self.retry_counts)
        np.save(os.path.join(self.save_path, "params.npy"), self.params)
        # save excluded data
        excluded_data_combined = xr.concat(self.excluded_data, dim='iter')
        print(excluded_data_combined)
        excluded_data_combined.to_netcdf(os.path.join(self.save_path, "excluded_data.nc"))
        
        # Convert config to JSON-serializable format
        config_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in self.config.__dict__.items()}
        
        with open(os.path.join(self.save_path, "config.json"), "w") as f:
            json.dump(config_dict, f)
        if self.track_learning:
            self.history.to_netcdf(os.path.join(self.save_path, "history.nc"))

def main():
    DATA_DIR = os.path.dirname(__file__) + "/../data/"
    
    # 設定の作成
    config = ExperimentConfig.create_default_config()
    
    # 実験の実行
    experiment = ExperimentManager(config, DATA_DIR)#, track_learning=True)
    experiment.run_experiment()
    experiment.save_results()

if __name__ == "__main__":
    main()