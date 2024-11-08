import json
import sys
import argparse
from pathlib import Path
import os
import numpy as np
import xarray as xr
from datetime import datetime



sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import metrics

BASE_DATA_DIR = os.path.dirname(__file__) +"/../data/"



def procece_data(folder_name):
    if folder_name == 'all':
        folder_names = os.listdir(BASE_DATA_DIR)
    else:
        folder_names = [folder_name]


    for folder_name in folder_names:
        try:
            DATA_DIR = os.path.join(BASE_DATA_DIR, folder_name)
            X = np.load(DATA_DIR+"/data.npy")
            params = np.load(DATA_DIR+"/params.npy", allow_pickle=True).item()
            retry_counts = np.load(DATA_DIR+"/retry_counts.npy")

            with open(os.path.join(DATA_DIR, "config.json"), "r") as f:
                config = json.load(f)
            if isinstance(config['c_alpha'][0], list):
                    
                n_components = params['W'].shape[1]
                n_simulations = params['W'].shape[0]
                n_dir_components = len(config['c_alpha'])
                dir_weights = config['weights'] if 'weights' in config else np.ones(n_dir_components) / n_dir_components
                evaluator = metrics.MixtureDirichletGaussianWishartEvaluator(n_components)
                results = []
        
            else:
                n_components = params['W'].shape[1]
                n_simulations = params['W'].shape[0]
                n_dir_components = 1
                dir_weights = np.array([1])
                evaluator = metrics.MixtureDirichletGaussianWishartEvaluator(n_components)
                results = []
                config['c_alpha'] = [config['c_alpha']]
            
            for si in range(n_simulations):
                NIW_params = [
                    {
                        'mu_0': params['m'][si][ci],
                        'kappa_0': params['beta'][si][ci],
                        'nu_0': params['nu'][si][ci],
                        'Psi_0': np.linalg.inv(params['W'][si][ci])
                    }
                    for ci in range(n_components)
                ]
                mixture_dirichlet_params = {
                    'weights': dir_weights,
                    'alphas': config['c_alpha'],
                }
                result = evaluator.expected_parameter_metrics(
                    NIW_params,
                    mixture_dirichlet_params,
                )
                results.append(result)
            def dict_to_xarray_direct(data_list):
                # データを変数ごとに分割
                var_names = ['expected_weights', 'variance_weights', 'expected_mahalanobis', 'variance_mahalanobis', 
                            'expected_overlap', 'variance_overlap', 'model_complexity', 'separation_confidence_interval']
                
                data_vars = {}
                for var in var_names:
                    values = [d[var] for d in data_list]
                    if var in ['expected_mahalanobis', 'variance_mahalanobis', 'expected_overlap', 'variance_overlap']:
                        data_vars[var] = (['simulation', 'component1', 'component2'], values)
                    elif var in ['expected_weights', 'variance_weights']:
                        data_vars[var] = (['simulation', 'component'], values)
                    else:
                        # data_vars[var] = (['simulation'], values)
                        pass
                
                coords = {
                    'simulation': np.arange(n_simulations),
                    'component': np.arange(n_components),
                    'component1': np.arange(n_components),
                    'component2': np.arange(n_components),
                }
                results = xr.Dataset(data_vars, coords=coords)
                return results
            results = dict_to_xarray_direct(results)
            # print(results)
            results.to_netcdf(os.path.join(DATA_DIR, "metrics.nc"))
        except Exception as e:
            print(e)
    print(config['c_alpha'])
    print(dir_weights)
    NIW_params = [
        {
            'mu_0': params['m'][si][ci],
            'kappa_0': params['beta'][si][ci],
            'nu_0': params['nu'][si][ci],
            'Psi_0': params['W'][si][ci],
        }
        for ci in range(n_components)
    ]
    mixture_dirichlet_params = {
        'weights': dir_weights,
        'alphas': config['c_alpha'],
    }
    result = evaluator.expected_parameter_metrics(
        NIW_params,
        mixture_dirichlet_params,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('folder_name', type=str, help='input file path')
    # parser.add_argument('--output', type=str, help='output file path')

    folder_name = parser.parse_args().folder_name
    procece_data(folder_name)