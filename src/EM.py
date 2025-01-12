import numpy as np


def initialize_parameters(data, num_components):
    np.random.seed(0)
    n, d = data.shape
    weights = np.ones(num_components) / num_components
    means = data[np.random.choice(n, num_components, False)]
    covariances = np.array([np.eye(d)] * num_components)
    return weights, means, covariances


def e_step(data, weights, means, covariances):
    n, d = data.shape
    num_components = len(weights)
    responsibilities = np.zeros((n, num_components))

    for k in range(num_components):
        diff = data - means[k]
        inv_cov = np.linalg.inv(covariances[k])
        term = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        responsibilities[:, k] = (
            weights[k]
            * np.exp(-0.5 * term)
            / np.sqrt(np.linalg.det(covariances[k]) * (2 * np.pi) ** d)
        )

    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


def m_step(data, responsibilities):
    n, d = data.shape
    num_components = responsibilities.shape[1]
    n_k = responsibilities.sum(axis=0)
    weights = n_k / n
    means = np.einsum("ij,ik->jk", responsibilities, data) / n_k[:, None]
    covariances = np.zeros((num_components, d, d))

    for k in range(num_components):
        diff = data - means[k]
        # covariances[k] = np.einsum('ij,ik,il->jkl', responsibilities[:, k], diff, diff) / n_k[k]
        covariances[k] = (
            np.einsum("i,ij,ik->jk", responsibilities[:, k], diff, diff) / n_k[k]
        )

    return weights, means, covariances


def log_likelihood(data, weights, means, covariances):
    n, d = data.shape
    num_components = len(weights)
    likelihood = 0

    for k in range(num_components):
        diff = data - means[k]
        inv_cov = np.linalg.inv(covariances[k])
        term = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        likelihood += np.sum(
            np.log(weights[k])
            - 0.5
            * (term + np.log(np.linalg.det(covariances[k])) + d * np.log(2 * np.pi))
        )

    return likelihood


def em_algorithm(data, num_components, max_iter=100, tol=1e-4):
    weights, means, covariances = initialize_parameters(data, num_components)
    log_likelihood_prev = None

    for i in range(max_iter):
        responsibilities = e_step(data, weights, means, covariances)
        weights, means, covariances = m_step(data, responsibilities)
        log_likelihood_curr = log_likelihood(data, weights, means, covariances)

        if (
            log_likelihood_prev is not None
            and abs(log_likelihood_curr - log_likelihood_prev) < tol
        ):
            break

        log_likelihood_prev = log_likelihood_curr

    return weights, means, covariances


# データの生成
np.random.seed(0)
data = np.vstack(
    [
        np.random.randn(100, 2) + np.array([3, 3]),
        np.random.randn(100, 2) + np.array([-3, -3]),
    ]
)

# EMアルゴリズムの実行
weights, means, covariances = em_algorithm(data, 3)

print("重み:", weights)
print("平均:", means)
print("共分散行列:", covariances)
