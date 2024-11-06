import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gammaln, digamma
from scipy.linalg import inv, det
from itertools import product


class MixtureDirichletGaussianWishartEvaluator:
    """
    混合ディリクレ分布とガウス・ウィシャート分布による混合ガウス分布の評価指標を計算するクラス
    """
    def __init__(self, n_components):
        self.n_components = n_components
    
    def expected_parameter_metrics(self, NIW_params, mixture_dirichlet_params):
        """
        ガウス・ウィシャート分布と混合ディリクレ分布のパラメータから評価指標を計算
        
        Parameters:
        -----------
        NIW_params : list of dict
            各成分のガウス・ウィシャート分布のパラメータ
            各辞書は以下のキーを持つ:
            - mu_0: 平均の事前分布の平均 (d次元ベクトル)
            - kappa_0: 平均の事前分布の精度
            - nu_0: ウィシャート分布の自由度
            - Psi_0: ウィシャート分布のスケール行列 (d×d行列)
            
        mixture_dirichlet_params : dict
            混合ディリクレ分布のパラメータ
            - weights: 各ディリクレ成分の混合重み (M次元)
            - alphas: 各ディリクレ成分のパラメータ (M × K次元)
            
        Returns:
        --------
        dict : 評価指標を含む辞書
        """
        metrics = {}
        
        # 混合ディリクレ分布のパラメータを展開
        dir_weights = mixture_dirichlet_params['weights']
        dir_alphas = mixture_dirichlet_params['alphas']
        n_dir_components = len(dir_weights)
        
        # 1. 重みの期待値と分散の計算
        exp_weights = np.zeros(self.n_components)
        var_weights = np.zeros(self.n_components)
        
        for m in range(n_dir_components):
            alpha_m = dir_alphas[m]
            alpha_0_m = np.sum(alpha_m)
            # m番目のディリクレ成分での期待値
            exp_weights_m = alpha_m / alpha_0_m
            # m番目のディリクレ成分での分散
            var_weights_m = (alpha_m * (alpha_0_m - alpha_m)) / (alpha_0_m**2 * (alpha_0_m + 1))
            
            # 混合重みで重み付けして加算
            exp_weights += dir_weights[m] * exp_weights_m
            var_weights += dir_weights[m] * (var_weights_m + exp_weights_m**2)
        
        var_weights -= exp_weights**2  # 全分散の法則による補正
        
        # 2. 重みの不確実性を考慮した評価指標の計算
        
        # 2.1 期待マハラノビス距離行列とその分散
        exp_mahalanobis = np.zeros((self.n_components, self.n_components))
        var_mahalanobis = np.zeros((self.n_components, self.n_components))
        
        for i, j in product(range(self.n_components), range(self.n_components)):
            if i < j:
                params_i = NIW_params[i]
                params_j = NIW_params[j]
                
                # 平均差の期待値
                mean_diff = params_i['mu_0'] - params_j['mu_0']
                d = mean_diff.shape[0]
                
                # 共分散の期待値の逆行列
                # print(params_i['Psi_0'])
                # print(params_i['nu_0'])
                exp_cov_sum = (params_i['Psi_0'] / (params_i['nu_0'] - d - 1) + 
                             params_j['Psi_0'] / (params_j['nu_0'] - d - 1)) / 2
                
                # マハラノビス距離の期待値
                exp_dist = np.sqrt(mean_diff.T @ inv(exp_cov_sum) @ mean_diff)
                exp_mahalanobis[i,j] = exp_dist
                exp_mahalanobis[j,i] = exp_dist
                
                # マハラノビス距離の分散（重みの不確実性を考慮）
                base_var = (1/params_i['kappa_0'] + 1/params_j['kappa_0']) * np.trace(inv(exp_cov_sum))
                weight_var = var_weights[i] + var_weights[j]
                total_var = base_var + weight_var * exp_dist**2
                
                var_mahalanobis[i,j] = total_var
                var_mahalanobis[j,i] = total_var
        
        # 2.2 重みを考慮した重なり係数の計算
        exp_overlap = np.zeros((self.n_components, self.n_components))
        var_overlap = np.zeros((self.n_components, self.n_components))
        
        for i, j in product(range(self.n_components), range(self.n_components)):
            if i < j:
                params_i = NIW_params[i]
                params_j = NIW_params[j]
                mean_diff = params_i['mu_0'] - params_j['mu_0']
                d = mean_diff.shape[0]
                
                # 各ディリクレ成分での重なり係数を計算
                overlap_components = []
                for m in range(n_dir_components):
                    alpha_m = dir_alphas[m]
                    weight_factor = (alpha_m[i] * alpha_m[j]) / (np.sum(alpha_m)**2)
                    
                    exp_cov_sum = (params_i['Psi_0'] / (params_i['nu_0'] - d - 1) + 
                                 params_j['Psi_0'] / (params_j['nu_0'] - d - 1))
                    
                    overlap = self._compute_overlap(mean_diff, exp_cov_sum, weight_factor)
                    overlap_components.append(dir_weights[m] * overlap)
                
                # 期待重なり係数
                exp_overlap_ij = np.sum(overlap_components)
                exp_overlap[i,j] = exp_overlap_ij
                exp_overlap[j,i] = exp_overlap_ij
                
                # 重なり係数の分散
                var_overlap_ij = self._compute_overlap_variance(
                    mean_diff, exp_cov_sum, params_i, params_j,
                    dir_weights, dir_alphas, i, j
                )
                var_overlap[i,j] = var_overlap_ij
                var_overlap[j,i] = var_overlap_ij
        
        # 2.3 モデルの複雑さ指標（重みの不確実性を考慮）
        complexity_scores = []
        for m in range(n_dir_components):
            alpha_m = dir_alphas[m]
            weight_uncertainty = -np.sum(digamma(alpha_m)) + digamma(np.sum(alpha_m))
            
            for k in range(self.n_components):
                params = NIW_params[k]
                d = params['mu_0'].shape[0]
                
                # パラメータの複雑さ
                param_complexity = (
                    np.log(det(params['Psi_0'])) -
                    d * np.log(params['kappa_0']) +
                    np.sum(digamma((params['nu_0'] - np.arange(d))/2))
                )
                
                complexity_scores.append(dir_weights[m] * (weight_uncertainty + param_complexity))
        
        metrics.update({
            'expected_weights': exp_weights,
            'variance_weights': var_weights,
            'expected_mahalanobis': exp_mahalanobis,
            'variance_mahalanobis': var_mahalanobis,
            'expected_overlap': exp_overlap,
            'variance_overlap': var_overlap,
            'model_complexity': np.mean(complexity_scores)
        })
        
        # 2.4 分離度の信頼区間（混合ディリクレ分布を考慮）
        separation_distributions = []
        for m in range(n_dir_components):
            alpha_m = dir_alphas[m]
            for i, j in product(range(self.n_components), range(self.n_components)):
                if i < j:
                    weight_factor = np.sqrt((alpha_m[i] * alpha_m[j]) / (np.sum(alpha_m)**2))
                    sep_mean = exp_mahalanobis[i,j] * weight_factor
                    sep_var = var_mahalanobis[i,j] * weight_factor**2
                    
                    separation_distributions.append({
                        'mean': sep_mean,
                        'std': np.sqrt(sep_var),
                        'weight': dir_weights[m]
                    })
        
        # 混合分布としての信頼区間を計算
        separation_means = np.array([d['mean'] for d in separation_distributions])
        separation_stds = np.array([d['std'] for d in separation_distributions])
        separation_weights = np.array([d['weight'] for d in separation_distributions])
        
        mean_separation = np.sum(separation_means * separation_weights)
        var_separation = np.sum((separation_stds**2 + separation_means**2) * separation_weights) - mean_separation**2
        
        metrics['separation_confidence_interval'] = (
            mean_separation - 1.96 * np.sqrt(var_separation),
            mean_separation + 1.96 * np.sqrt(var_separation)
        )
        
        return metrics
    
    def _compute_overlap(self, mean_diff, cov_sum, weight_factor):
        """重なり係数を計算"""
        d = mean_diff.shape[0]
        exp_term = -0.5 * mean_diff.T @ inv(cov_sum) @ mean_diff
        return weight_factor * np.exp(exp_term) / np.sqrt((2*np.pi)**d * det(cov_sum))
    
    def _compute_overlap_variance(self, mean_diff, cov_sum, params_i, params_j, 
                                dir_weights, dir_alphas, i, j):
        """重なり係数の分散を計算"""
        d = mean_diff.shape[0]
        total_var = 0
        
        for m in range(len(dir_weights)):
            alpha_m = dir_alphas[m]
            alpha_0_m = np.sum(alpha_m)
            
            # ディリクレ成分での重みの不確実性
            weight_var = (alpha_m[i] * alpha_m[j] * (alpha_0_m - alpha_m[i] - alpha_m[j])) / \
                        (alpha_0_m**3 * (alpha_0_m + 1))
            
            # パラメータの不確実性
            param_var = (1/params_i['kappa_0'] + 1/params_j['kappa_0']) * \
                       np.trace(inv(cov_sum)) + \
                       (2/(params_i['nu_0']-d-1) + 2/(params_j['nu_0']-d-1)) * \
                       (d + np.trace(inv(cov_sum) @ inv(cov_sum)))
            
            overlap = self._compute_overlap(mean_diff, cov_sum, 1)
            total_var += dir_weights[m] * overlap**2 * (weight_var + param_var)
            
        return total_var