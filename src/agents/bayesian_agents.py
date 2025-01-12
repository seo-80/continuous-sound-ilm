import numpy as np
import xarray as xr
from scipy.special import digamma, gammaln, gamma

def logB(W, nu):
    D = W.shape[-1]
    return D * np.log(2) + D * digamma(nu/2) - nu/2 * np.linalg.slogdet(W)[1]

def logC(alpha):
    return gammaln(alpha.sum()) - gammaln(alpha).sum()

def multi_student_t(X, m, L, nu):
    D = X.shape[1]
    # Convert xarray DataArray to numpy array if needed
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(m, 'values'):
        m = m.values
    if hasattr(L, 'values'):
        L = L.values
    if hasattr(nu, 'values'):
        nu = nu.values
    if len(m.shape) == 1:
        m = m.reshape(1, -1)
        
    diff = X - m
    log_part1 = gammaln((nu + D)/2)
    log_part2 = -gammaln(nu/2)
    log_part3 = -D/2 * np.log(nu*np.pi)
    log_part4 = -0.5 * np.log(np.linalg.det(L))
    
    # Ensure all inputs are numpy arrays for einsum
    diff = np.asarray(diff)
    L_inv = np.linalg.inv(L)
    quadratic_term = np.einsum("nj,jk,nk->n", diff, L_inv, diff)
    log_part5 = -(nu+D)/2 * np.log(1 + 1/nu * quadratic_term)

    # Calculate the log of the final result
    log_result = log_part1 + log_part2 + log_part3 + log_part4 + log_part5

    # Convert the log result back to a regular number
    result = np.exp(log_result)

    return result

def filter_high_entropy(data, model, args):
    threshold = args["threshold"]
    p = model.predict_proba(data)
    p = np.clip(p, 1e-10, 1-1e-10)
    entropy = -np.sum(p * np.log(p), axis=1)
    return entropy < threshold

def filter_low_max_prob(data, model, args):
    threshold = args["threshold"]
    p = model.predict_proba(data)
    max_prob = np.max(p, axis=1)
    return max_prob > threshold

def filter_missunderstand(data, model, args):
    p = model.predict_proba(data)
    listener_perception = np.argmax(p, axis=1)
    speaker_perception = data["Z"].argmax(dim="k").values
    return listener_perception == speaker_perception

FILTER_DICT = {
    "high_entropy": filter_high_entropy,
    "low_max_prob": filter_low_max_prob,
    "missunderstand": filter_missunderstand,
    "none": None
}

class BayesianGaussianMixtureModel:
    def __init__(self, K, D, alpha0, beta0, nu0, m0, W0, c_alpha, pi_mixture_ratio=None, fit_filter=None, fit_filter_args=None, generate_filter=None, generate_filter_args=None, track_learning=False):
        self.params = xr.Dataset({
            'K': K,
            'D': D,
            'alpha0': (['k'], alpha0 if isinstance(alpha0, np.ndarray) else alpha0 * np.ones(K)),
            'beta0': (['k'], beta0 if isinstance(beta0, np.ndarray) else beta0 * np.ones(K)),
            'nu0': (['k'], nu0 if isinstance(nu0, np.ndarray) else nu0 * np.ones(K)),
            'm0': (['k', 'd'], m0 if m0.shape == (K, D) else np.tile(m0, (K, 1))),
            'W0': (['k', 'd', 'd'], [xr.DataArray(W0, dims=['d', 'd']) for _ in range(K)]),
            'c_alpha': (['k'] if c_alpha.ndim == 1 else ['c', 'k'], c_alpha if isinstance(c_alpha, np.ndarray) else c_alpha * np.ones(K)),
            'pi_mixture_ratio': (['c'], pi_mixture_ratio if pi_mixture_ratio is not None else np.ones(self.comopnent_num)/self.comopnent_num) if isinstance(c_alpha, np.ndarray) and c_alpha.shape[0] != K else None
        })

        self.data = xr.Dataset()
        self.state = xr.Dataset({
            'alpha': (['k'], np.zeros(K)),
            'beta': (['k'], np.zeros(K)),
            'nu': (['k'], np.zeros(K)),
            'm': (['k', 'd'], np.zeros((K, D))),
            'W': (['k', 'd', 'd'], np.zeros((K, D, D)))
        })
        self.lower_bound = None
        self._init_params()

        self.fit_filter = FILTER_DICT.get(fit_filter, fit_filter)
        self.generate_filter = FILTER_DICT.get(generate_filter, generate_filter)
        self.fit_filter_args = fit_filter_args
        self.generate_filter_args = generate_filter_args
        self.track_learning = track_learning
        if self.track_learning:
            self.history = xr.Dataset()
        self.excluded_data = []

    def _init_params(self, X=None, random_state=None):
        if X is None:
            N, D = 0, self.params['D'].values
        else:
            N, D = X.shape
        rnd = np.random.RandomState(seed=random_state)

        self.state['alpha'] = self.params['alpha0'] + N / self.params['K'].values * np.ones(self.params['K'].values)
        self.state['beta'] = self.params['beta0'] + N / self.params['K'].values * np.ones(self.params['K'].values)
        self.state['nu'] = self.params['nu0'] + N / self.params['K'].values * np.ones(self.params['K'].values)
        self.state['m'] = self.params['m0'] + rnd.randn(self.params['K'].values, self.params['D'].values)
        print(self.params['W0'].values.shape)
        print(self.state['W'].values.shape)
        self.state['W'] = xr.DataArray(
            self.params['W0'].values,
            dims=['k', 'd', 'd'],
            coords={
                'k': np.arange(self.params['K'].values),
                'd': np.arange(self.params['D'].values)
            }
        )

    def _e_like_step(self, X):
        N, _ = X.shape

        if self.params['c_alpha'] is None:
            tpi = np.exp(digamma(self.state['alpha']) - digamma(self.state['alpha'].sum()))
        else:
            if self.params['pi_mixture_ratio'] is not None:
                tpi = np.sum(self.params['c_alpha'], axis=0) / np.sum(self.params['c_alpha'])
            else:
                tpi = self.params['c_alpha'] / np.sum(self.params['c_alpha'])

        arg_digamma = self.state['nu'].values.reshape(self.params['K'].values, 1) - np.arange(0, self.params['D'].values, 1).reshape(1, self.params['D'].values)
        tlam = np.exp(digamma(arg_digamma / 2).sum(axis=1) + self.params['D'].values * np.log(2) + np.log(np.linalg.det(self.state['W'])))

        diff = X.values.reshape(N, 1, self.params['D'].values) - self.state['m'].values.reshape(1, self.params['K'].values, self.params['D'].values)
        exponent = self.params['D'].values / self.state['beta'] + self.state['nu'] * np.einsum("nkj,nkj->nk", np.einsum("nki,kij->nkj", diff, self.state['W']), diff)

        exponent_subtracted = exponent - np.reshape(exponent.min(axis=1), (N, 1))
        rho = tpi * np.sqrt(tlam) * np.exp(-0.5 * exponent_subtracted)
        r = rho / np.reshape(rho.sum(axis=1), (N, 1))

        return r

    def _m_like_step(self, X, r):
        N, _ = X.shape
        n_samples_in_component = r.sum(axis=0)
        barx = r.T @ X / np.reshape(n_samples_in_component, (self.params['K'].values, 1))
        diff = X.values.reshape(N, 1, self.params['D'].values) - barx.reshape(1, self.params['K'].values, self.params['D'].values)
        S = np.einsum("nki,nkj->kij", np.einsum("nk,nki->nki", r, diff), diff) / np.reshape(n_samples_in_component, (self.params['K'].values, 1, 1))

        self.state['alpha'] = self.params['alpha0'] + n_samples_in_component
        self.state['beta'] = self.params['beta0'] + n_samples_in_component
        self.state['nu'] = self.params['nu0'] + n_samples_in_component
        self.state['m'] = (self.params['beta0'] * self.params['m0'] + barx * np.reshape(n_samples_in_component, (self.params['K'].values, 1))) / np.reshape(self.state['beta'], (self.params['K'].values, 1))

        diff2 = barx - self.params['m0']
        Winv = np.reshape(np.linalg.inv(self.params['W0']), (1, self.params['D'].values, self.params['D'].values)) + \
            S * np.reshape(n_samples_in_component, (self.params['K'].values, 1, 1)) + \
            np.reshape(self.params['beta0'] * n_samples_in_component / (self.params['beta0'] + n_samples_in_component), (self.params['K'].values, 1, 1)) * np.einsum("ki,kj->kij", diff2, diff2)
        self.state['W'] = xr.DataArray(
            np.linalg.inv(Winv),
            dims=['k', 'd', 'd'],
            coords={
                'k': np.arange(self.params['K'].values),
                'd': np.arange(self.params['D'].values)
            }
        )

    def _calc_lower_bound(self, r):
        r = np.clip(r, 1e-10, 1-1e-10)
        return - (r * np.log(r)).sum() + \
            logC(self.params['alpha0']) - logC(self.state['alpha']) + \
            self.params['D'].values / 2 * (np.log(self.params['beta0']).sum() - np.log(self.state['beta']).sum()) + \
            self.params['K'].values * logB(self.params['W0'], self.params['nu0']).sum() - logB(self.state['W'], self.state['nu']).sum()

    def fit(self, data, max_iter=1e3, tol=1e-4, random_state=None, disp_message=False):
        if self.fit_filter is not None and not self.fit_filter(data, self, self.fit_filter_args):
            return False
        if self.data.sizes.get('n', 0) == 0:
            self.data = data
            self._init_params(self.data.X, random_state=random_state)
        else:
            self.data = xr.concat([self.data, data], dim='n')

        r = self._e_like_step(self.data.X)
        lower_bound = self._calc_lower_bound(r)

        for i in range(max_iter):
            self._m_like_step(self.data.X, r)
            r = self._e_like_step(self.data.X)

            lower_bound_prev = lower_bound
            lower_bound = self._calc_lower_bound(r)

            if abs(lower_bound - lower_bound_prev) < tol:
                break

        self.lower_bound = lower_bound

        if disp_message:
            print(f"n_iter : {i}")
            print(f"convergend : {i < max_iter}")
            print(f"lower bound : {lower_bound}")
            print(f"Change in the variational lower bound : {lower_bound - lower_bound_prev}")

        return True

    def fit_from_agent(self, source_agent, N, max_iter=1e3, tol=1e-4, random_state=None, disp_message=False):
        X = source_agent.generate(N)
        self.fit(X, max_iter=max_iter, tol=tol, random_state=random_state, disp_message=disp_message)

    def _predict_joint_proba(self, X):
        L = np.reshape((self.state['nu'] + 1 - self.params['D'].values) * self.state['beta'] / (1 + self.state['beta']), (self.params['K'].values, 1, 1)) * self.state['W']
        tmp = np.zeros((len(X), self.params['K'].values))
        for k in range(self.params['K'].values):
            tmp[:, k] = multi_student_t(X, self.state['m'][k], L[k], self.state['nu'][k] + 1 - self.params['D'].values)
        return tmp * np.reshape(self.state['alpha'] / self.state['alpha'].sum(), (1, self.params['K'].values))

    def calc_prob_density(self, X):
        joint_proba = self._predict_joint_proba(X)
        return joint_proba.sum(axis=1)

    def predict_proba(self, data):
        if isinstance(data, xr.Dataset):
            X = data.X.values
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
        else:
            X = data
        joint_proba = self._predict_joint_proba(X)
        return joint_proba / joint_proba.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def generate(self, n_samples):
        if self.params['c_alpha'] is None:
            alpha_norm = self.state['alpha'] / self.state['alpha'].sum()
            z_new = np.random.multinomial(1, alpha_norm, size=n_samples)
        else:
            if self.params['pi_mixture_ratio'] is not None:
                comopnent_idx = np.random.choice(self.params['c_alpha'].shape[0], size=n_samples, p=self.params['pi_mixture_ratio'].values)
                z_new = []
                for i in range(n_samples):
                    alpha_norm = self.params['c_alpha'][comopnent_idx[i]] / np.sum(self.params['c_alpha'][comopnent_idx[i]])
                    z_new.append(np.random.multinomial(1, alpha_norm, size=1))
                z_new = np.vstack(z_new)
            else:
                alpha_norm = self.params['c_alpha'] / np.sum(self.params['c_alpha'])
                z_new = np.random.multinomial(1, alpha_norm, size=n_samples)
        X_new = np.zeros((n_samples, self.params['D'].values))

        for k in range(self.params['K'].values):
            idx = np.where(z_new[:, k] == 1)[0]
            if len(idx) > 0:
                X_new[idx] = np.random.multivariate_normal(
                    self.state['m'][k],
                    np.linalg.inv(self.state['beta'][k] * self.state['W'][k]),
                    size=len(idx)
                )

        ret_ds = xr.Dataset(
            {
                'X': (['n', 'd'], X_new),
            },
            coords={'n': np.arange(n_samples), 'd': np.arange(self.params['D'].values)}
        )
        return ret_ds
    
    @property
    def X(self):
        return self.data.X
    @property
    def C(self):
        return self.data.C
    @property
    def Z(self):
        return self.data.Z
    @property
    def alpha(self):
        return self.state.alpha
    @property
    def beta(self):
        return self.state.beta
    @property
    def nu(self):
        return self.state.nu
    @property
    def m(self):
        return self.state.m
    @property
    def W(self):
        return self.state.W
    

class BayesianGaussianMixtureModelWithContext(BayesianGaussianMixtureModel):
    def __init__(self, K, D, alpha0, beta0, nu0, m0, W0, c_alpha, pi_mixture_ratio=None, fit_filter=None, fit_filter_args=None, generate_filter=None, generate_filter_args=None, track_learning=False):
        super().__init__(K, D, alpha0, beta0, nu0, m0, W0, c_alpha, pi_mixture_ratio, fit_filter, fit_filter_args, generate_filter, generate_filter_args, track_learning)
        self.data['C'] = (['n', 'k'], np.zeros((0, self.params['K'].values)))
        self.data['Z'] = (['n', 'k'], np.zeros((0, self.params['K'].values)))

    def fit(self, data, max_iter=1e3, tol=1e-4, random_state=None, disp_message=False):
        if self.fit_filter is not None and not self.fit_filter(data, self, self.fit_filter_args):
            return False
        if self.data.sizes.get('n', 0) == 0:
            self.data = data
            self._init_params(self.data.X, random_state=random_state)
        else:
            self.data = xr.concat([self.data, data], dim='n')

        unique_values, counts = np.unique(self.data.X, axis=0, return_counts=True)
        duplicate_mask = counts > 1
        duplicate_values_num = counts[duplicate_mask].sum()
        if duplicate_values_num > 0:
            print("Warning: Duplicated values in X are detected.")
            print(f"Number of duplicated values in X: {duplicate_values_num}/{len(self.data.X)}")

        r = self._e_like_step(self.data.X.values, self.data.C.values)
        lower_bound = self._calc_lower_bound(r)

        for i in range(max_iter):
            self._m_like_step(self.data.X.values, r)
            r = self._e_like_step(self.data.X.values, self.data.C.values)

            lower_bound_prev = lower_bound
            lower_bound = self._calc_lower_bound(r)

            for k in range(self.params['K'].values):
                if not np.all(np.isfinite(self.state['W'][k].values)):
                    print(f"Warning: W[{k}] contains non-finite values")
                    self._init_params(self.data.X, random_state=random_state)

                eigvals = np.linalg.eigvals(self.state['W'][k])
                if np.any(eigvals > 1e10) or np.any(eigvals < -1e10):
                    print(f"Warning: W[{k}] has very large eigenvalues indicating potential divergence")
                    self._init_params(self.data.X, random_state=random_state)
            if abs(lower_bound - lower_bound_prev) < tol:
                break

        self.lower_bound = lower_bound

        if disp_message:
            print(f"n_iter : {i}")
            print(f"convergend : {i < max_iter}")
            print(f"lower bound : {lower_bound}")
            print(f"Change in the variational lower bound : {lower_bound - lower_bound_prev}")
        return True

    def fit_from_agent(self, source_agent, N, max_iter=1000, tol=0.0001, random_state=None, disp_message=False):
        data, excluded_data = source_agent.generate(N, return_excluded_data=True).values()
        self.excluded_data = excluded_data

        self.history = xr.Dataset({
            'alpha': (['n', 'k'], np.zeros((N, self.params['K'].values))),
            'beta': (['n', 'k'], np.zeros((N, self.params['K'].values))),
            'nu': (['n', 'k'], np.zeros((N, self.params['K'].values))),
            'm': (['n', 'k', 'd'], np.zeros((N, self.params['K'].values, self.params['D'].values))),
            'W': (['n', 'k', 'd', 'd'], np.zeros((N, self.params['K'].values, self.params['D'].values, self.params['D'].values)))
        }, coords={
            'n': np.arange(N),
            'k': np.arange(self.params['K'].values),
            'd': np.arange(self.params['D'].values)
        })

        if self.fit_filter is None and self.track_learning is False:
            self.fit(data, max_iter=max_iter, tol=tol, random_state=random_state, disp_message=disp_message)
        else:
            excluded_data_list = []
            count = -1
            for i in range(N):
                while True:
                    count += 1
                    if count >= N:
                        print("Warning: The number of excluded data exceeds the number of generated data.")
                        count = 0
                        data = source_agent.generate(N)
                    self._init_params(random_state=random_state)
                    if self.fit(data.sel(n=count), max_iter=max_iter, tol=tol, random_state=random_state, disp_message=disp_message):
                        if self.track_learning:
                            self.history['alpha'][i] = self.state['alpha']
                            self.history['beta'][i] = self.state['beta']
                            self.history['nu'][i] = self.state['nu']
                            self.history['m'][i] = self.state['m']
                            self.history['W'][i] = self.state['W']
                        break
                    else:
                        excluded_data_list.append(data.sel(n=count))
            if len(excluded_data_list) > 0:
                self.excluded_data = xr.concat(excluded_data_list, dim='n').assign_coords(n=np.arange(len(excluded_data_list)))
            else:
                self.excluded_data = xr.Dataset()

    def _e_like_step(self, X, C):
        N, _ = X.shape

        # Convert inputs to numpy arrays to avoid xarray dimension issues
        tpi = C.values if hasattr(C, 'values') else C
        nu = self.state['nu'].values if hasattr(self.state['nu'], 'values') else self.state['nu']
        beta = self.state['beta'].values if hasattr(self.state['beta'], 'values') else self.state['beta']
        W = self.state['W'].values if hasattr(self.state['W'], 'values') else self.state['W']
        m = self.state['m'].values if hasattr(self.state['m'], 'values') else self.state['m']
        D = self.params['D'].values if hasattr(self.params['D'], 'values') else self.params['D']

        arg_digamma = nu.reshape(self.params['K'].values, 1) - np.arange(0, D, 1).reshape(1, D)
        tlam = np.exp(digamma(arg_digamma / 2).sum(axis=1) + D * np.log(2) + np.log(np.linalg.det(W)))

        diff = X.reshape(N, 1, D) - m.reshape(1, self.params['K'].values, D)
        exponent = D / beta + nu * np.einsum("nkj,nkj->nk", np.einsum("nki,kij->nkj", diff, W), diff)

        exponent_subtracted = exponent - np.reshape(exponent.min(axis=1), (N, 1))
        rho = tpi * np.sqrt(tlam) * np.exp(-0.5 * exponent_subtracted)
        r = rho / np.reshape(rho.sum(axis=1), (N, 1))

        return r

    def _m_like_step(self, X, r):
        N, _ = X.shape
        n_samples_in_component = r.sum(axis=0)
        barx = r.T @ X / np.reshape(n_samples_in_component, (self.params['K'].values, 1))
        diff = X.reshape(N, 1, self.params['D'].values) - barx.reshape(1, self.params['K'].values, self.params['D'].values)
        S = np.einsum("nki,nkj->kij", np.einsum("nk,nki->nki", r, diff), diff) / np.reshape(n_samples_in_component, (self.params['K'].values, 1, 1))

        self.state['alpha'] = self.params['alpha0'] + n_samples_in_component
        self.state['beta'] = self.params['beta0'] + n_samples_in_component
        self.state['nu'] = self.params['nu0'] + n_samples_in_component
        self.state['m'] = (self.params['beta0'] * self.params['m0'] + barx * np.reshape(n_samples_in_component, (self.params['K'].values, 1))) / np.reshape(self.state['beta'].values, (self.params['K'].values, 1))

        diff2 = barx - self.params['m0']
        Winv = np.reshape(np.linalg.inv(self.params['W0']), (-1, self.params['D'].values, self.params['D'].values)) + \
            S * np.reshape(n_samples_in_component, (self.params['K'].values, 1, 1)) + \
            np.reshape((self.params['beta0'] * n_samples_in_component / (self.params['beta0'] + n_samples_in_component)).values, (self.params['K'].values, 1, 1)) * np.einsum("ki,kj->kij", diff2, diff2)
        self.state['W'] = xr.DataArray(
            np.linalg.inv(Winv),
            dims=['k', 'd', 'd'],
            coords={
                'k': np.arange(self.params['K'].values),
                'd': np.arange(self.params['D'].values)
            }
        )

    def generate(self, n_samples, return_excluded_data=False):
        collected_datasets = []
        excluded_data = []
        n_filtered_samples = 0

        while n_filtered_samples < n_samples:
            batch_size = min(n_samples - n_filtered_samples, n_samples)

            if self.params['c_alpha'].values[()] is None:
                alpha_norm = self.state['alpha'] / self.state['alpha'].sum()
                z_new = np.random.multinomial(1, alpha_norm, size=batch_size)
                C_new = np.random.dirichlet(self.params['c_alpha'], size=batch_size)
            else:
                if self.params['pi_mixture_ratio'].values[()] is not None:
                    comopnent_idx = np.random.choice(self.params['c_alpha'].shape[0], size=batch_size, p=self.params['pi_mixture_ratio'].values)
                    z_new = []
                    C_new = []
                    for i in range(batch_size):
                        C_new_temp = np.random.dirichlet(self.params['c_alpha'][comopnent_idx[i]], size=1)[0]
                        C_new.append(C_new_temp)
                        z_new.append(np.random.multinomial(1, C_new_temp, size=1))
                    z_new = np.vstack(z_new)
                    C_new = np.vstack(C_new)
                else:
                    C_new_temp = np.random.dirichlet(self.params['c_alpha'], size=batch_size)
                    z_new = np.array([np.random.multinomial(1, C_new_temp[i], size=1)[0] for i in range(batch_size)])
                    C_new = C_new_temp

            X_new = np.zeros((batch_size, self.params['D'].values))
            for k in range(self.params['K'].values):
                idx = np.where(z_new[:, k] == 1)[0]
                if len(idx) > 0:
                    if not np.all(np.isfinite(self.state['W'][k].values)):
                        raise ValueError("W must be finite.")
                    if not np.all(np.isfinite(self.state['m'][k].values)):
                        raise ValueError("m must be finite.")

                    min_eig = np.min(np.linalg.eigvals(self.state['W'][k]))
                    if min_eig < 0:
                        self.state['W'][k] -= 10 * min_eig * np.eye(self.params['D'].values)

                    X_new[idx] = np.random.multivariate_normal(
                        self.state['m'][k],
                        np.linalg.inv(self.state['beta'][k].values * self.state['W'][k].values),
                        size=len(idx)
                    )
            temp_ret_ds = xr.Dataset(
                {
                    'X': (['n', 'd'], X_new),
                    'C': (['n', 'k'], C_new),
                    'Z': (['n', 'k'], z_new),
                },
                coords={
                    'n': np.arange(n_filtered_samples, n_filtered_samples + batch_size),
                    'd': np.arange(self.params['D'].values),
                    'k': np.arange(self.params['K'].values)
                }
            )

            temp_excluded_data = None
            if self.generate_filter is not None:
                filtered_index = self.generate_filter(temp_ret_ds, self, self.generate_filter_args)
                temp_excluded_data = temp_ret_ds.isel(n=~filtered_index)
                temp_ret_ds = temp_ret_ds.isel(n=filtered_index)
            if temp_excluded_data is None:
                temp_excluded_data = xr.Dataset(
                    {
                        'X': (['n', 'd'], np.zeros((0, self.params['D'].values))),
                        'C': (['n', 'k'], np.zeros((0, self.params['K'].values))),
                        'Z': (['n', 'k'], np.zeros((0, self.params['K'].values))),
                    },
                    coords={'n': [], 'd': np.arange(self.params['D'].values), 'k': np.arange(self.params['K'].values)}
                )

            if len(temp_ret_ds.X) > 0:
                temp_n_samples = min(len(temp_ret_ds.X), n_samples - n_filtered_samples)
                if temp_n_samples > 0:
                    collected_datasets.append(temp_ret_ds.isel(n=slice(0, temp_n_samples)))
                    if return_excluded_data:
                        excluded_data.append(temp_excluded_data)
                    n_filtered_samples += temp_n_samples

        if collected_datasets:
            final_ds = xr.concat(collected_datasets, dim='n')
            final_ds = final_ds.assign_coords(n=np.arange(len(final_ds.n)))

            if return_excluded_data:
                if excluded_data:
                    final_excluded_data = xr.concat(excluded_data, dim='n')
                    final_excluded_data = final_excluded_data.assign_coords(n=np.arange(len(final_excluded_data.n)))
                else:
                    final_excluded_data = xr.Dataset(
                        {
                            'X': (['n', 'd'], np.zeros((0, self.params['D'].values))),
                            'C': (['n', 'k'], np.zeros((0, self.params['K'].values))),
                            'Z': (['n', 'k'], np.zeros((0, self.params['K'].values))),
                        },
                        coords={'n': [], 'd': np.arange(self.params['D'].values), 'k': np.arange(self.params['K'].values)}
                    )

                return {
                    'data': final_ds,
                    'excluded_data': final_excluded_data
                }
            else:
                return final_ds

    def predict_proba(self, data):
        if isinstance(data, tuple):
            X, C = data
        elif isinstance(data, xr.Dataset):
            X = data.X.values
            C = data.C.values
            if len(X.shape) == 1:
                X, C = X.reshape(1, -1), C.reshape(1, -1)
        else:
            X = data
        joint_proba = self._predict_joint_proba(X, C)
        return joint_proba / joint_proba.sum(axis=1).reshape(-1, 1)

    def _predict_joint_proba(self, X, C):
        L = np.reshape(((self.state['nu'] + 1 - self.params['D'].values) * self.state['beta'] / (1 + self.state['beta'])).values, (self.params['K'].values, 1, 1)) * self.state['W'].values
        tmp = np.zeros((len(X), self.params['K'].values))
        for k in range(self.params['K'].values):
            tmp[:, k] = multi_student_t(X, self.state['m'][k], L[k], self.state['nu'][k] + 1 - self.params['D'].values)
        return tmp * C
