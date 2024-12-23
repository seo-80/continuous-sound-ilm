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
import hashlib
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agents import BayesianGaussianMixtureModel, BayesianGaussianMixtureModelWithContext

@dataclass
class ExperimentConfig:
    K: int  # 混合成分数
    D: int  # 次元数
    N: int  # サンプル数
    agent: str
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
        K = 4
        D = 2
        N = 500
        
        # Default parameters
        c_alpha = np.array([1/K for _ in range(K)])
        alpha0 = 100.0
        # beta0 = np.array([1 if i%2 ==0 else 0.01 for i in range(K)])
        beta0 = np.array([0.1 for i in range(K)])
        nu0 = D + 2.0
        m0_range = 0
        m0 = np.array([[m0_range*np.cos(2*np.pi*i/K), m0_range*np.sin(2*np.pi*i/K)] for i in range(K)])
        # m0 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        W0 = np.eye(D)*0.02

        return cls(
            K=K, D=D, N=N,
            agent="BayesianGaussianMixtureModelWithContext",
            alpha0=alpha0, beta0=beta0, nu0=nu0,
            c_alpha=c_alpha, m0=m0, W0=W0,
            iter=2,
            fit_filter_name="none",
            generate_filter_name="missunderstand",
            fit_filter_args={},
            generate_filter_args={},
        )
    
        # weght = 4
        # c_alpha = np.array([[2*(weght-1)/weght/K,2*(weght-1)/weght/K, 2/weght/K, 2/weght/K], [2/weght/K, 2/weght/K, 2*(weght-1)/weght/K,2*(weght-1)/weght/K,]])
        # alpha0 = 100.0
        # beta0 = np.array([1 if i%2 ==0 else 0.01 for i in range(K)])
        # beta0 = np.array([8 for i in range(K)])
        # nu0 = D + 2.0
        # m0_range = 10
        # m0 = np.array([[m0_range*np.cos(2*np.pi*i/K), m0_range*np.sin(2*np.pi*i/K)] for i in range(K)])
        # m0 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        # W0 = np.eye(D)*0.02

        # return cls(
        #     K=K, D=D, N=N, 
        #     agent="BayesianGaussianMixtureModelWithContext",
        #     alpha0=alpha0, beta0=beta0, nu0=nu0,
        #     c_alpha=c_alpha, m0=m0, W0=W0,
        #     iter=1000,
        #     fit_filter_name="none",
        #     generate_filter_name="missunderstand",
        #     fit_filter_args={},
        #     generate_filter_args={},
        # )
    @classmethod
    def load_config(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        # Convert lists to numpy arrays if they exist in config
        if isinstance(config["beta0"], list):
            config["beta0"] = np.array(config["beta0"])
        if isinstance(config["m0"], list):
            config["m0"] = np.array(config["m0"])
        if isinstance(config["W0"], list):
            config["W0"] = np.array(config["W0"])
        if isinstance(config["c_alpha"], list):
            config["c_alpha"] = np.array(config["c_alpha"])

        # !応急処置 configの修正
        m0_range = 5
        K = config["K"]
        m0 = np.array([[m0_range*np.cos(2*np.pi*i/K), m0_range*np.sin(2*np.pi*i/K)] for i in range(K)])
        # # # m0 = np.array([[4,5],[4,-5],[2,5],[2,-5],[0,5],[0,-5],[-2,5],[-2,-5]])
        config["m0"] = m0
        # beta0 = np.array([10 if i%2 ==0 else 1 for i in range(K)])
        # config["beta0"] = beta0
        # beta0 = np.array([10 for i in range(config["K"    ])])
        # config["beta0"] = beta0
        # config['generate_filter_name'] = "missunderstand"
        config['iter'] = 1000

        ret_config = cls(
            K=config["K"],
            D=config["D"],
            N=config["N"],
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
        )
            
        return ret_config

class ExperimentManager:
    def __init__(self, config: ExperimentConfig, save_dir: str, track_learning: bool = False):
        self.config = config
        self.save_dir = save_dir
        self.setup_data_directory()
        self.track_learning = track_learning
        print(config)
        
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
        # np.random.seed(0)
        true_K = self.config.K
        true_means = self.config.m0
        true_covars = np.array([np.eye(2) * 0.1 for _ in range(true_K)])
        true_alpha = np.array([1 for _ in range(true_K)])
        N=self.config.N
        D=2


        X_0 = np.zeros((N, D))
        C_0 = np.random.dirichlet(true_alpha, size=N)
        z_0 = np.array([np.random.multinomial(1, c) for c in C_0])
        
        for k in range(true_K):
            X_0[z_0[:, k] == 1] = np.random.multivariate_normal(
                true_means[k],
                true_covars[k],
                size=np.sum(z_0[:, k] == 1)
            )
        initial_data = xr.Dataset(
            {
                'X': (['n', 'd'], X_0),
                'C': (['n', 'k'], C_0),
                'Z': (['n', 'k'], z_0)
            },
            coords={
                'n': np.arange(N),
                'd': np.arange(self.config.D),
                'k': np.arange(self.config.K)
            }
        )
        
            
        return initial_data

    def create_agent(self, is_parent: bool = False, track_learning: bool = False) -> Any:
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
                track_learning=track_learning
            )
        elif self.config.agent == "BayesianGaussianMixtureModel":
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
        folder_name = self.save_path.split("/")[-1]
        folder_name_hash = hashlib.md5(folder_name.encode()).hexdigest()
        random_seed = int(folder_name_hash, 16) % (2**32)
        np.random.seed(random_seed)

        parent_agent = self.create_agent()


        for i in tqdm.tqdm(range(self.config.iter)):
            child_agent = self.create_agent(track_learning= (i == self.config.iter-1) if self.track_learning == "Final" else self.track_learning)
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
            if self.track_learning is True or self.track_learning == "Final":
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
        print(self.excluded_data)
        try:
            excluded_data_combined = xr.concat(self.excluded_data, dim='iter')
            print(excluded_data_combined)
            excluded_data_combined.to_netcdf(os.path.join(self.save_path, "excluded_data.nc"))
        except:
            print("error")
        
        # Convert config to JSON-serializable format
        config_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in self.config.__dict__.items()}
        
        with open(os.path.join(self.save_path, "config.json"), "w") as f:
            json.dump(config_dict, f)
        if self.track_learning:
            self.history.to_netcdf(os.path.join(self.save_path, "history.nc"))

def main(folder_name: str):
    DATA_DIR = os.path.dirname(__file__) + "/../data/"
    
    # 設定の作成
    if folder_name is not "None":
        config = ExperimentConfig.load_config(os.path.join(DATA_DIR, folder_name, "config.json"))
    else:
        config = ExperimentConfig.create_default_config()
    
    # # 実験の実行
    # experiment = ExperimentManager(config, DATA_DIR,track_learning=True)
    experiment = ExperimentManager(config, DATA_DIR,track_learning="Final")
    experiment = ExperimentManager(config, DATA_DIR)
    experiment.run_experiment()
    experiment.save_results()
    print(experiment.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name',nargs="?" , type=str, default="None", help='input file path')
    folder_name = parser.parse_args().folder_name
    main(folder_name)
