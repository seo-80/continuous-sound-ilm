import numpy as np
import os
import json
import xarray as xr
import sys
import argparse
from pathlib import Path






sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import metrics






#load data
DATA_DIR = os.path.dirname(__file__) +"/../data/"
folder_names = os.listdir(DATA_DIR)

# 検索要件
def all_elements_equal(actual_value, target_value) -> bool:
    """
    Check if all elements in the iterable are equal.
    
    Parameters:
    iterable (Iterable): An iterable containing elements to be checked.
    
    Returns:
    bool: True if all elements are equal, False otherwise.
    """
    actual_value = np.array(actual_value)
    if target_value is None:
        target_value = actual_value[0]
    return np.all(actual_value == target_value)

def all_elements_not_equal(actual_value, target_value) -> bool:
    """
    Check if all elements in the iterable are not equal.
    
    Parameters:
    iterable (Iterable): An iterable containing elements to be checked.
    
    Returns:
    bool: True if all elements are not equal, False otherwise.
    """
    return not all_elements_equal(actual_value, target_value)

def is_shape_equal(data, expected_shape) -> bool:
    """
    Check if the dimension of the data matches the expected dimension.
    
    Parameters:
    data (Union[np.ndarray, List]): The data to be checked.
    expected_dim (int): The expected dimension.
    
    Returns:
    bool: True if the dimension matches, False otherwise.
    """
    data = np.array(data)

    return data.shape == expected_shape


K = 8
conditions = [
    {
        "path": "agent",
        "operator": "eq",
        "value": "BayesianGaussianMixtureModelWithContext"
    },
    {
        "path": "K",
        "operator": "eq",
        "value": K
    },
    # {
    #     "path": "filter_name",
    #     "operator": "eq",
    #     "value": "high_entropy"
    # }
    {
        "path": "c_alpha",
        "operator": all_elements_not_equal,
        "value": None
    },
    {
        "path": "c_alpha",
        "operator": is_shape_equal,
        "value": (K,)
    }

]


    # if config mathces, add folder_name to results
from typing import Any, List, Dict, Union, Callable
import operator
from collections.abc import Iterable

class JsonSearcher:
    """
    JSONデータを柔軟に検索するためのユーティリティクラス
    """
    def __init__(self, data: Dict):
        self.data = data
        
    def _get_nested_value(self, keys: str) -> Any:
        """
        ドット区切りのキーパスから値を取得
        例: "true_means.0.1" -> true_means配列の最初の要素の2番目の値
        """
        curr = self.data
        for key in keys.split('.'):
            try:
                if key.isdigit():
                    curr = curr[int(key)]
                else:
                    curr = curr[key]
            except (KeyError, IndexError, TypeError):
                return None
        return curr

    def search(self, conditions: List[Dict]) -> bool:
        """
        複数の検索条件に基づいて検索を実行
        
        conditions: [
            {
                "path": "キーパス（ドット区切り）",
                "operator": "比較演算子",
                "value": "比較値"
            },
            ...
        ]
        """
        for condition in conditions:
            path = condition["path"]
            op = condition["operator"]
            target_value = condition["value"]
            
            actual_value = self._get_nested_value(path)
            if actual_value is None:
                return False
                
            if callable(op):
                op_func = op
            
            else:
                op_func = {
                    "eq": operator.eq,
                    "ne": operator.ne,
                    "gt": operator.gt,
                    "ge": operator.ge,
                    "lt": operator.lt,
                    "le": operator.le,
                    "in": lambda x, y: x in y,
                    "contains": lambda x, y: y in x if isinstance(x, (str, list, dict)) else False,
                    "type": lambda x, y: isinstance(x, eval(y)),
                    "len": lambda x, y: len(x) == y if isinstance(x, Iterable) else False
                }.get(op)
            
            if not op_func:
                raise ValueError(f"Unknown operator: {op}")
                
                
            if not op_func(actual_value, target_value):
                return False
                
        return True

    def find_all_paths(self, curr_path: str = "", curr_obj: Any = None) -> List[str]:
        """
        JSONデータ内のすべてのパスを取得
        """
        if curr_obj is None:
            curr_obj = self.data
            
        paths = []
        
        if isinstance(curr_obj, dict):
            for key, value in curr_obj.items():
                new_path = f"{curr_path}.{key}" if curr_path else key
                paths.append(new_path)
                paths.extend(self.find_all_paths(new_path, value))
                
        elif isinstance(curr_obj, list):
            for i, value in enumerate(curr_obj):
                new_path = f"{curr_path}.{i}" if curr_path else str(i)
                paths.append(new_path)
                paths.extend(self.find_all_paths(new_path, value))
                
        return paths

# 使用例


if __name__ == "__main__":
    configs = []
    for folder_name in folder_names:
        if os.path.exists(os.path.join(DATA_DIR, folder_name, "config.json")):
            config = json.load(open(os.path.join(DATA_DIR, folder_name, "config.json")))
            configs.append({
                'folder_name': folder_name,
                'config': config
            })
    for i in range(len(configs)):
        parser = argparse.ArgumentParser()
        parser.add_argument('--print_config', action='store_true')
        print_config = parser.parse_args().print_config
        searcher = JsonSearcher(configs[i]['config'])
        # 検索の実行
        search_result = searcher.search(conditions)
        
        if search_result:
            print(configs[i]['folder_name'])
            if print_config:
                print(configs[i]['config'])
        

    