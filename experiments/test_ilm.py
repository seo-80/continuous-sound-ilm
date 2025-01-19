import argparse
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
import hashlib
import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agents import BayesianGMM, BayesianGMMWithContext, BayesianGMMWithContextWithAttenuation

print('start')
@dataclass
class BaseExperimentConfig(ABC):
    # 共通のパラメータ
    K: int  # 混合成分数
    D: int  # 次元数
    N: int  # サンプル数
    agent: str
    iter: int
    fit_filter_name: str = "none"
    generate_filter_name: str = "none"
    fit_filter_args: Dict[str, Any] = field(default_factory=dict)
    generate_filter_args: Dict[str, Any] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    def convert_lists_to_arrays(self) -> None:
        """Convert all list attributes to numpy arrays"""
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            elif isinstance(value, dict):
                # Convert lists inside dictionaries
                for k, v in value.items():
                    if isinstance(v, list):
                        value[k] = np.array(v)
    
    @abstractmethod
    def validate(self) -> bool:

        """設定の妥当性を検証する"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in self.__dict__.items()
        }

@dataclass
class BayesianGMMConfig(BaseExperimentConfig):
    # BayesianGMM固有のパラメータ
    alpha0: float = None
    beta0: Optional[np.ndarray] = None
    nu0: Optional[float] = None
    m0: Optional[np.ndarray] = None
    W0: Optional[np.ndarray] = None
    
    def validate(self) -> bool:
        if self.agent != "BayesianGMM":
            return False
        if any(param is None for param in [self.alpha0, self.beta0, self.nu0, self.m0, self.W0]):
            return False
        return True

    @classmethod
    def create_default_config(cls) -> "BayesianGMMConfig":
        K = 4
        D = 2
        N = 500

        # Default parameters
        alpha0 = 100.0
        beta0 = np.array([0.1 for i in range(K)])
        nu0 = D + 2.0
        m0_range = 0
        m0 = np.array([
            [m0_range * np.cos(2 * np.pi * i / K), m0_range * np.sin(2 * np.pi * i / K)]
            for i in range(K)
        ])
        W0 = np.eye(D) * 0.02

        return cls(
            K=K, D=D, N=N,
            agent="BayesianGMM",
            alpha0=alpha0,
            beta0=beta0,
            nu0=nu0,
            m0=m0,
            W0=W0,
            iter=10
        )

@dataclass
class BayesianGMMWithContextConfig(BayesianGMMConfig):
    # BayesianGMMWithContext固有のパラメータ
    c_alpha: Optional[np.ndarray] = None
    
    def validate(self) -> bool:
        if self.agent != "BayesianGMMWithContext":
            return False
        if not super().validate() or self.c_alpha is None:
            return False
        return True


    @classmethod
    def create_default_config(cls) -> "BayesianGMMWithContextConfig":
        K = 4
        D = 2
        N = 500

        # Default parameters
        c_alpha = np.array([1/K for _ in range(K)])
        alpha0 = 100.0
        beta0 = np.array([0.1 for i in range(K)])
        nu0 = D + 2.0
        m0_range = 0
        m0 = np.array([
            [m0_range * np.cos(2 * np.pi * i / K), m0_range * np.sin(2 * np.pi * i / K)]
            for i in range(K)
        ])
        W0 = np.eye(D) * 0.02

        return cls(
            K=K, D=D, N=N,
            agent="BayesianGMMWithContext",
            alpha0=alpha0,
            beta0=beta0,
            nu0=nu0,
            c_alpha=c_alpha,
            m0=m0,
            W0=W0,
            iter=10
        )
@ dataclass
class BayesianGMMWithContextWithAttenuationConfig(BayesianGMMWithContextConfig):
    # BayesianGMMWithContextWithAttenuation固有のパラメータ
    S: int = None
    s_alpha0: Optional[np.ndarray] = None
    context_mix_ratio: Optional[float] = None
    
    def validate(self) -> bool:
        if self.agent != "BayesianGMMWithContextWithAttenuation":
            return False
        if not super().validate() or self.s_alpha0 is None or self.context_mix_ratio is None:
            return False
        return True

    @classmethod
    def create_default_config(cls) -> "BayesianGMMWithContextWithAttenuationConfig":
        K = 4
        D = 2
        N = 500

        # Default parameters
        c_alpha = np.array([1/K for _ in range(K)])
        alpha0 = 100.0
        beta0 = np.array([0.1 for i in range(K)])
        nu0 = D + 2.0
        m0_range = 0
        m0 = np.array([
            [m0_range * np.cos(2 * np.pi * i / K), m0_range * np.sin(2 * np.pi * i / K)]
            for i in range(K)
        ])
        W0 = np.eye(D) * 0.02
        s_alpha0 = np.ones((K, 1))
        context_mix_ratio = 0.5

        return cls(
            K=K, D=D, N=N,
            agent="BayesianGMMWithContextWithAttenuation",
            alpha0=alpha0,
            beta0=beta0,
            nu0=nu0,
            c_alpha=c_alpha,
            m0=m0,
            W0=W0,
            s_alpha0=s_alpha0,
            context_mix_ratio=context_mix_ratio,
            iter=10
        )

# ファクトリー関数
def create_config(config_dict: Dict[str, Any]) -> BaseExperimentConfig:
    agent_type = config_dict["agent"]
    if agent_type == "BayesianGMM":
        return BayesianGMMConfig(**config_dict)
    elif agent_type == "BayesianGMMWithContext":
        return BayesianGMMWithContextConfig(**config_dict)
    elif agent_type == "BayesianGMMWithContextWithAttenuation":
        return BayesianGMMWithContextWithAttenuationConfig(**config_dict)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

class ExperimentManager:
    def __init__(self, config: BaseExperimentConfig, save_dir: str, track_learning: bool = False, save_as_dataset: bool = True):
        self.config = config
        self.save_dir = save_dir
        self.setup_data_directory()
        self.track_learning = track_learning
        self.save_as_dataset = save_as_dataset

        # Initialize storage for results
        self.params = {
            "alpha": np.zeros((config.iter,) + (config.K,)),
            "beta": np.zeros((config.iter,) + (config.K,)),
            "nu": np.zeros((config.iter,) + (config.K,)),
            "m": np.zeros((config.iter, config.K, config.D)),
            "W": np.zeros((config.iter, config.K, config.D, config.D)),
        }

        if save_as_dataset:
            # Store everything in a single xarray Dataset
            self.data = xr.Dataset()
            self.params = xr.Dataset()
        else:
            # Store as separate lists/arrays
            self.X: List[np.ndarray] = []
            self.C: List[np.ndarray] = []
            self.Z: List[np.ndarray] = []
        self.excluded_data = []
        self.retry_counts = []

        if self.track_learning:
            self.history = xr.Dataset(
                {
                    "alpha": (["iter", "n", "k"], np.zeros((config.iter, config.N, config.K))),
                    "beta": (["iter", "n", "k"], np.zeros((config.iter, config.N, config.K))),
                    "nu": (["iter", "n", "k"], np.zeros((config.iter, config.N, config.K))),
                    "m": (["iter", "n", "k", "d"], np.zeros((config.iter, config.N, config.K, config.D))),
                    "W": (["iter", "n", "k", "d1", "d2"], np.zeros((config.iter, config.N, config.K, config.D, config.D))),
                },
                coords={
                    "iter": np.arange(config.iter),
                    "n": np.arange(config.N),
                    "K": np.arange(config.K),
                    "d": np.arange(config.D),
                    "d1": np.arange(config.D),
                    "d2": np.arange(config.D),
                }
            )
    def setup_data_directory(self):
        """実験データ保存用のディレクトリを設定"""
        folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = os.path.join(self.save_dir, folder_name)
        os.makedirs(self.save_path, exist_ok=True)

    def create_agent(self, is_parent: bool = False, track_learning: bool = False) -> Any:
        """エージェントの作成"""
        agent_params = {
            "K": self.config.K,
            "D": self.config.D,
            "alpha0": self.config.alpha0,
            "beta0": self.config.beta0,
            "nu0": self.config.nu0,
            "m0": self.config.m0,
            "W0": self.config.W0,
            "track_learning": track_learning
        }

        if self.config.agent == "BayesianGMMWithContext":
            agent_params.update({
                "c_alpha": self.config.c_alpha,
                "fit_filter": "none" if is_parent else self.config.fit_filter_name,
                "fit_filter_args": self.config.fit_filter_args,
                "generate_filter": "none" if is_parent else self.config.generate_filter_name,
                "generate_filter_args": self.config.generate_filter_args,
            })
            return BayesianGMMWithContext(**agent_params)
        elif self.config.agent == "BayesianGMM":
            return BayesianGMM(**agent_params)
        elif self.config.agent == "BayesianGMMWithContextWithAttenuation":
            agent_params.update({
                "S": self.config.S,
                "s_alpha0": self.config.s_alpha0,
                "c_alpha": self.config.c_alpha,
                "context_mix_ratio": self.config.context_mix_ratio,
                "fit_filter": "none" if is_parent else self.config.fit_filter_name,
                "fit_filter_args": self.config.fit_filter_args,
                "generate_filter": "none" if is_parent else self.config.generate_filter_name,
                "generate_filter_args": self.config.generate_filter_args,
            })
            return BayesianGMMWithContextWithAttenuation(**agent_params)
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")

    def run_experiment(self):
        """実験の実行"""
        folder_name = self.save_path.split("/")[-1]
        folder_name_hash = hashlib.md5(folder_name.encode()).hexdigest()
        random_seed = int(folder_name_hash, 16) % (2**32)
        np.random.seed(random_seed)

        parent_agent = self.create_agent()

        for i in tqdm.tqdm(range(self.config.iter)):
            child_agent = self.create_agent(
                track_learning=(i == self.config.iter - 1) if self.track_learning == "Final" else self.track_learning
            )
            retry_count = []
            child_agent.fit_from_agent(parent_agent, N=self.config.N)
            if self.save_as_dataset:
                self.data = xr.concat([self.data, child_agent.data], dim="iter")
                self.params = xr.concat([self.params, child_agent.state], dim="iter")
            else:
                self.X.append(child_agent.X)
                self.Z.append(child_agent.Z)
                if isinstance(self.config, BayesianGMMWithContextConfig):
                    self.C.append(child_agent.C)
                # パラメータの保存
                self.params["alpha"][i] = child_agent.alpha
                self.params["beta"][i] = child_agent.beta
                self.params["nu"][i] = child_agent.nu
                self.params["m"][i] = child_agent.m
                self.params["W"][i] = child_agent.W
            self.retry_counts.append(retry_count)

            self.excluded_data.append(child_agent.excluded_data)

            if self.track_learning is True or self.track_learning == "Final":
                self.history["alpha"][i] = child_agent.history["alpha"]
                self.history["beta"][i] = child_agent.history["beta"]
                self.history["nu"][i] = child_agent.history["nu"]
                self.history["m"][i] = child_agent.history["m"]
                self.history["W"][i] = child_agent.history["W"]

            parent_agent = child_agent

    def save_results(self):
        """結果の保存"""
        if self.save_as_dataset:
            self.data.to_netcdf(os.path.join(self.save_path, "data.nc"))
            self.params.to_netcdf(os.path.join(self.save_path, "params.nc"))
        else:
            np.save(os.path.join(self.save_path, "data.npy"), self.X)
            if isinstance(self.config, BayesianGMMWithContextConfig):
                np.save(os.path.join(self.save_path, "context.npy"), self.C)
                np.save(os.path.join(self.save_path, "Z.npy"), self.Z)
            np.save(os.path.join(self.save_path, "params.npy"), self.params)
        np.save(os.path.join(self.save_path, "retry_counts.npy"), self.retry_counts)

        # save excluded data
        try:
            excluded_data_combined = xr.concat(self.excluded_data, dim="iter")
            excluded_data_combined.to_netcdf(os.path.join(self.save_path, "excluded_data.nc"))
        except:
            print("Error saving excluded data")

        # Save config
        with open(os.path.join(self.save_path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)

        if self.track_learning:
            self.history.to_netcdf(os.path.join(self.save_path, "history.nc"))

def main(folder_name: str):
    DATA_DIR = os.path.dirname(__file__) + "/../data/"

    # 設定の作成
    if folder_name != "None":
        if folder_name.endswith('.json'):
            with open(folder_name, 'r') as f:
                config_dict = json.load(f)
            config = create_config(config_dict)
        else:
            with open(os.path.join(DATA_DIR, folder_name, "config.json"), 'r') as f:
                config_dict = json.load(f)
            config = create_config(config_dict)
    else:
        config = BayesianGMMWithContextConfig.create_default_config()
    config.convert_lists_to_arrays()
    # 実験の実行
    experiment = ExperimentManager(config, DATA_DIR)
    experiment.run_experiment()
    experiment.save_results()
    print(experiment.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_name", nargs="?", type=str, default="None", help="input file path")
    folder_name = parser.parse_args().folder_name
    main(folder_name)