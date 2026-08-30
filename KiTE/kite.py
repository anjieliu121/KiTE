import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import vmap, jit
from typing import Dict, List, Optional, Sequence, Tuple

# -------------------------
# Kernel and KLCE Functions
# -------------------------

def rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X and Y.

    This function computes the Gaussian (radial basis function) kernel
    elementwise between two datasets X and Y using kernel coefficient gamma.

    Parameters
    ----------
    X : jnp.ndarray
        First data array of shape (n_samples, n_features) or (n_samples,) for single feature.
    Y : jnp.ndarray
        Second data array of the same shape as X.
    gamma : float
        Kernel coefficient, typically defined as 1 / (sigma^2).

    Returns
    -------
    jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape. Got {X.shape} and {Y.shape}.")
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    squared_diff = (
        jnp.sum(X**2, axis=1)[:, None]
        + jnp.sum(Y**2, axis=1)[None, :]
        - 2 * jnp.dot(X, Y.T)
    )
    return jnp.exp(-gamma * squared_diff)


@jit
def create_kernel(
    X: jnp.ndarray,
    p: jnp.ndarray,
    prob_kernel_width: float,
    x_kernel_width: float
) -> jnp.ndarray:
    """
    Create combined kernel matrix for KLCE.

    This function computes the elementwise product of two RBF kernels:
    one over predicted probabilities p and one over feature matrix X.

    Parameters
    ----------
    X : jnp.ndarray
        Feature matrix of shape (n_samples, n_features).
    p : jnp.ndarray
        Predicted probabilities array of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth parameter for the probability kernel.
    x_kernel_width : float
        Bandwidth parameter for the feature kernel.

    Returns
    -------
    jnp.ndarray
        Combined kernel matrix of shape (n_samples, n_samples).
    """
    if p.ndim != 1:
        raise ValueError(f"p must be a 1D array. Got ndim={p.ndim}.")
    if X.shape[0] != p.shape[0]:
        raise ValueError(
            f"Number of samples in X and p must match. Got {X.shape[0]} and {p.shape[0]}."
        )

    p_reshaped = p.reshape(-1, 1)
    gamma_p = 1.0 / (prob_kernel_width ** 2)
    gamma_x = 1.0 / (x_kernel_width ** 2)

    K_pp = rbf_kernel(p_reshaped, p_reshaped, gamma_p)
    K_xx = rbf_kernel(X, X, gamma_x)

    return K_pp * K_xx


@jit
def KLCE2_estimator(K: jnp.ndarray, err: jnp.ndarray) -> float:
    """
    Compute the KLCE2 estimator from kernel matrix K and error vector err.

    This function sums the off-diagonal elements of the elementwise product
    between K and the outer product of err with itself, normalized by n*(n-1).

    Parameters
    ----------
    K : jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    err : jnp.ndarray
        Error vector of shape (n_samples,).

    Returns
    -------
    float
        KLCE2 estimator value.
    """
    if K.ndim != 2:
        raise ValueError(f"K must be a 2D array. Got ndim={K.ndim}.")
    if err.ndim != 1:
        raise ValueError(f"err must be a 1D array. Got ndim={err.ndim}.")
    if K.shape[0] != K.shape[1] or K.shape[0] != err.shape[0]:
        raise ValueError(
            f"Shape mismatch: K must be square of size n and err length n. Got K.shape={K.shape}, err.shape={err.shape}."
        )

    err_outer = jnp.outer(err, err)
    K_err = K * err_outer

    mask = jnp.ones_like(K_err) - jnp.eye(K_err.shape[0])
    K_err_off_diag = K_err * mask

    n = err.shape[0]
    return jnp.sum(K_err_off_diag) / (n * (n - 1))


def KLCE2_boosting(
    f: jnp.ndarray,
    X_cal: jnp.ndarray,
    y: jnp.ndarray,
    prob_kernel_width: float,
    x_kernel_width: float
) -> float:
    """
    Compute the KLCE2 boosting loss for calibration.

    This function calculates the KLCE2 estimator using the base predictions f,
    calibration features X_cal, true labels y, and given kernel widths.

    Parameters
    ----------
    f : jnp.ndarray
        Base prediction probabilities of shape (n_samples,).
    X_cal : jnp.ndarray
        Calibration feature matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        True labels of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth for the probability kernel.
    x_kernel_width : float
        Bandwidth for the feature kernel.

    Returns
    -------
    float
        KLCE2 boosting loss.
    """
    p_err = y - f
    K = create_kernel(X_cal, f, prob_kernel_width, x_kernel_width)
    return KLCE2_estimator(K, p_err)


@jit
def KLCE2_null_estimator(err: jnp.ndarray, K: jnp.ndarray, key: jnp.ndarray) -> float:
    """
    Compute a single null sample for the KLCE2 test by permutation.

    This function permutes the error vector err and computes the KLCE2 estimator
    against the kernel matrix K for a single random key.

    Parameters
    ----------
    err : jnp.ndarray
        Error vector of shape (n_samples,).
    K : jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    key : jnp.ndarray
        PRNG key for permutation.

    Returns
    -------
    float
        KLCE2 estimator for permuted errors.
    """
    idx = random.permutation(key, len(err))
    return KLCE2_estimator(K, err[idx])


def compute_null_distribution(
    p_err: jnp.ndarray,
    K: jnp.ndarray,
    key: jnp.ndarray,
    iterations: int
) -> jnp.ndarray:
    """
    Compute the null distribution of KLCE2 estimators over multiple permutations.

    Parameters
    ----------
    p_err : jnp.ndarray
        Error vector of shape (n_samples,).
    K : jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    key : jnp.ndarray
        PRNG key for permutation splitting.
    iterations : int
        Number of null samples to generate.

    Returns
    -------
    jnp.ndarray
        Array of null KLCE2 estimates of length `iterations`.
    """
    vmapped_null = jit(vmap(KLCE2_null_estimator, (None, None, 0)))
    keys = random.split(key, iterations)
    return vmapped_null(p_err, K, keys)


def KLCE_test(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    p: jnp.ndarray,
    prob_kernel_width: float,
    iterations: int,
    key: jnp.ndarray,
    x_kernel_width: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Perform the KLCE hypothesis test comparing model predictions to true labels.

    This function computes the test statistic and p-value by comparing the observed
    KLCE2 estimator against a null distribution generated by permutations.

    Parameters
    ----------
    X : jnp.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : jnp.ndarray
        True label vector of shape (n_samples,).
    p : jnp.ndarray
        Predicted probability vector of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth for the probability kernel.
    iterations : int
        Number of permutations for null distribution.
    key : jnp.ndarray
        PRNG key for random operations.
    x_kernel_width : float, optional
        Bandwidth for the feature kernel. If omitted, ``prob_kernel_width``
        is used for both kernels.

    Returns
    -------
    Tuple[float, float]
        test_value : Observed KLCE2 statistic.
        p_value : Corresponding p-value (at least ``1 / iterations``).
    """
    if x_kernel_width is None:
        x_kernel_width = prob_kernel_width
    K = create_kernel(X, p, prob_kernel_width, x_kernel_width)
    p_err = Y - p
    test_value = KLCE2_estimator(K, p_err)
    resolution = 1.0 / iterations
    test_null = compute_null_distribution(p_err, K, key, iterations)
    p_value = jnp.maximum(resolution, resolution * jnp.sum(test_null > test_value))
    return test_value, p_value

# ------------------------------------
# Recalibration MLP Model using Optax
# ------------------------------------

def init_recalibrated_model_params(
    rng: jnp.ndarray,
    layer_sizes: Sequence[int],
    scale: float = 1e-1
) -> Dict[str, jnp.ndarray]:
    """
    Initialize parameters for the recalibration MLP model.

    This function creates weight matrices and bias vectors for each layer
    based on `layer_sizes`, scaled by `scale` and initialized from a normal distribution.

    Parameters
    ----------
    rng : jnp.ndarray
        PRNG key for parameter initialization.
    layer_sizes : Sequence[int]
        Sizes of each layer including input and output dimensions.
    scale : float, optional
        Scaling factor for random initialization. Default is 1e-1.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to initialized arrays.
    """
    keys = random.split(rng, 2 * (len(layer_sizes) - 1))
    params: Dict[str, jnp.ndarray] = {}
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
        W_key, b_key = keys[2*i], keys[2*i + 1]
        params[f"W{i}"] = scale * random.normal(W_key, (in_dim, out_dim))
        params[f"b{i}"] = scale * random.normal(b_key, (out_dim,))
    return params


def recalibrated_model_apply(
    params: Dict[str, jnp.ndarray],
    x: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply the recalibration MLP model to input features.

    This function performs a forward pass through the MLP layers using ReLU
    activations, producing a correction term for the base probabilities.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Model parameters mapping layer names to weight and bias arrays.
    x : jnp.ndarray
        Input feature array of shape (n_samples, n_features).

    Returns
    -------
    jnp.ndarray
        Model output of shape (n_samples, 1).
    """
    num_layers = len(params) // 2
    h = x
    for i in range(num_layers):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        h = jnp.dot(h, W) + b
        if i < num_layers - 1:
            h = jax.nn.relu(h)
    return h

class recalibrated_model:
    """
    Recalibration model combining distillation loss and KLCE penalty.

    This class implements a simple MLP-based recalibration of base probabilities
    trained to minimize a combination of KL divergence and kernel-based calibration error.
    """

    def __init__(
        self,
        sigma_k: float = 0.1,
        sigma_l: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        num_steps: int = 1000,
        learning_rate: float = 0.001,
        hidden_layer_sizes: Tuple[int, ...] = (64, 64),
        seed: int = 121
    ) -> None:
        """
        Initialize hyperparameters for the recalibration model.

        Parameters
        ----------
        sigma_k : float, optional
            Kernel width for probability kernel. Default is 0.1.
        sigma_l : float, optional
            Kernel width for feature kernel. Default is 1.0.
        alpha : float, optional
            Weight for the distillation loss term. Default is 0.5.
        beta : float, optional
            Weight for the KLCE penalty term. Default is 0.5.
        num_steps : int, optional
            Number of training steps. Default is 1000.
        learning_rate : float, optional
            Optimizer learning rate. Default is 0.001.
        hidden_layer_sizes : Tuple[int, ...], optional
            Sizes of hidden MLP layers. Default is (64, 64).
        seed : int, optional
            Random seed for initialization. Default is 121.
        """
        self.sigma_k = sigma_k
        self.sigma_l = sigma_l
        self.alpha = alpha
        self.beta = beta
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.seed = seed
        self.params: Optional[Dict[str, jnp.ndarray]] = None
        self.loss_history: Optional[List[float]] = None

    def total_loss(
        self,
        params: Dict[str, jnp.ndarray],
        base_probs: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray
    ) -> float:
        """
        Compute the total loss combining distillation and KLCE penalty.

        This function calculates the KL divergence between base_probs and
        recalibrated predictions, then adds the kernel-based calibration error.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            Recalibration model parameters.
        base_probs : jnp.ndarray
            Original predicted probabilities of shape (n_samples,).
        x : jnp.ndarray
            Calibration features of shape (n_samples, n_features).
        y : jnp.ndarray
            True labels of shape (n_samples,).

        Returns
        -------
        float
            Weighted sum of distillation loss and KLCE penalty.
        """
        n = base_probs.shape[0]
        features = jnp.column_stack([jnp.ones(n), base_probs, x])
        correction = recalibrated_model_apply(params, features).squeeze()
        f_recalibrated = base_probs + correction
        f_recalibrated = jnp.clip(f_recalibrated, 1e-6, 1.0 - 1e-6)
        base_probs_stable = jnp.clip(base_probs, 1e-6, 1.0 - 1e-6)
        log_ratio1 = jnp.log(jnp.maximum(base_probs_stable / f_recalibrated, 1e-10))
        log_ratio2 = jnp.log(
            jnp.maximum((1 - base_probs_stable) / (1 - f_recalibrated), 1e-10)
        )
        kl_div = base_probs_stable * log_ratio1 + (1 - base_probs_stable) * log_ratio2
        distill_loss = jnp.mean(kl_div)
        x_2d = x[:, None] if x.ndim == 1 else x
        klce_loss = KLCE2_boosting(
            f_recalibrated, x_2d, y, self.sigma_k, self.sigma_l
        )
        return self.alpha * distill_loss + self.beta * klce_loss

    def fit(
        self,
        y_proba: jnp.ndarray,
        x_cal: jnp.ndarray,
        y: jnp.ndarray
    ) -> None:
        """
        Train the recalibration model on calibration data.

        This method optimizes model parameters to minimize the total loss
        over specified number of steps using the Adam optimizer.

        Parameters
        ----------
        y_proba : jnp.ndarray
            Base probability predictions of shape (n_samples,).
        x_cal : jnp.ndarray
            Calibration feature matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            True labels of shape (n_samples,).

        Returns
        -------
        None
        """
        rng = random.PRNGKey(self.seed)
        n = y_proba.shape[0]
        if x_cal.ndim == 1:
            x_cal = x_cal[:, None]
        input_dim = 2 + x_cal.shape[1]
        layer_sizes = [input_dim] + list(self.hidden_layer_sizes) + [1]
        params = init_recalibrated_model_params(rng, layer_sizes)
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        @jit
        def step(
            params: Dict[str, jnp.ndarray],
            opt_state: optax.OptState
        ) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, float]:
            loss_val, grads = jax.value_and_grad(self.total_loss)(
                params, y_proba, x_cal, y
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        loss_history: List[float] = []
        for i in range(self.num_steps):
            params, opt_state, loss_val = step(params, opt_state)
            loss_history.append(float(loss_val))
            if i % 100 == 0:
                print(f"Step {i}: total loss = {loss_val:.6f}")
        self.params = params
        self.loss_history = loss_history

    def predict_proba(
        self,
        y_proba: jnp.ndarray,
        x_cal: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate recalibrated probability predictions.

        This method applies the trained recalibration model to new data,
        returning corrected probability estimates.

        Parameters
        ----------
        y_proba : jnp.ndarray
            Base probability predictions of shape (n_samples,).
        x_cal : jnp.ndarray
            Calibration features of shape (n_samples, n_features).

        Returns
        -------
        jnp.ndarray
            Recalibrated probability vector of shape (n_samples,).
        """
        if self.params is None:
            raise RuntimeError("Call fit() before predict_proba().")
        m = y_proba.shape[0]
        if x_cal.ndim == 1:
            x_cal = x_cal[:, None]
        features_new = jnp.column_stack([jnp.ones(m), y_proba, x_cal])
        correction = recalibrated_model_apply(self.params, features_new).squeeze()
        f_recalibrated_new = y_proba + correction
        return jnp.clip(f_recalibrated_new, 1e-6, 0.999999)

    def get_labels(
        self,
        y_proba: jnp.ndarray,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Convert probability predictions to binary labels.

        Parameters
        ----------
        y_proba : jnp.ndarray
            Probability vector of shape (n_samples,).
        threshold : float, optional
            Classification threshold. Default is 0.5.

        Returns
        -------
        List[int]
            Binary labels (0 or 1) for each sample.
        """
        return [1 if y > threshold else 0 for y in y_proba]

    def accuracy_score(
        self,
        y_pred: Sequence[int],
        y: Sequence[int]
    ) -> float:
        """
        Compute the classification accuracy.

        Parameters
        ----------
        y_pred : Sequence[int]
            Predicted labels of shape (n_samples,).
        y : Sequence[int]
            True labels of shape (n_samples,).

        Returns
        -------
        float
            Proportion of correctly classified samples.
        """
        predictions = jnp.array(y_pred)
        actual_labels = jnp.array(y)
        return float(jnp.mean(predictions == actual_labels))
