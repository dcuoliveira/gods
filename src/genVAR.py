import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

def generate_ar_process(A_list, sigma_list, T, p, coef_switch_points=None, cov_switch_points=None, seed=None):
    """
    Generate an AR(p) process with separate regime changes for coefficients and covariance matrices.
    
    Parameters:
    A_list : list of np.array
        List of coefficient matrices (p x k x k) for different regimes.
    sigma_list : list of np.array
        List of covariance matrices (k x k) for different regimes.
    T : int
        Number of time steps.
    p : int
        Order of the autoregressive process.
    coef_switch_points : list of int, optional
        List of time steps where the coefficient matrix switches.
    cov_switch_points : list of int, optional
        List of time steps where the covariance matrix switches.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns:
    X : np.array
        Simulated time series data of shape (T, k).
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = A_list[0].shape[1]
    X = np.zeros((T, k))
    
    # Initialize regime pointers
    coef_index, cov_index = 0, 0
    coef_switch_points = [0] + (coef_switch_points if coef_switch_points else [T])
    cov_switch_points = [0] + (cov_switch_points if cov_switch_points else [T])
    
    # Generate AR process
    for t in range(p, T):
        # Update regime if needed
        if t >= coef_switch_points[min(coef_index + 1, len(coef_switch_points) - 1)]:
            coef_index = min(coef_index + 1, len(A_list) - 1)
        if t >= cov_switch_points[min(cov_index + 1, len(cov_switch_points) - 1)]:
            cov_index = min(cov_index + 1, len(sigma_list) - 1)
        
        # Draw noise and update process
        epsilon = np.random.multivariate_normal(mean=np.zeros(k), cov=sigma_list[cov_index])
        X[t] = np.einsum('pij, pj -> i', A_list[coef_index], X[t-p:t][::-1]) + epsilon
    
    return X

# Example usage
if __name__ == "__main__":
    k = 2
    T = 100
    p = 2

    A1 = np.array([[[0.5, 0.2], [0.1, 0.3]], [[0.3, 0.1], [0.2, 0.4]]])
    A2 = np.array([[[0.2, 0.4], [0.4, 0.2]], [[0.2, 0.2], [0.1, 0.3]]])
    A3 = np.array([[[0.3, -0.1], [-0.2, 0.5]], [[0.1, -0.2], [0.3, 0.1]]])
    
    sigma1 = np.array([[0.5, 0.3], [0.3, 0.1]])
    sigma2 = np.array([[3, 0.1], [0.1, 2]])
    sigma3 = np.array([[0.2, -0.2], [-0.2, 0.2]])
    
    # No switch points example
    X_no_switch = generate_ar_process([A1], [sigma1], T, p, coef_switch_points=None, cov_switch_points=None, seed=42)
    
    # Example with change points
    coef_switch_points = None
    cov_switch_points = [T // 3, 2*T // 3]
    X_with_switch = generate_ar_process([A2], 
                                        [sigma1, sigma2, sigma3], 
                                        T, 
                                        p, 
                                        coef_switch_points, 
                                        cov_switch_points, 
                                        seed=42)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    for i in range(k):
        plt.plot(X_with_switch[:, i], label=f'X{i+1}')
    plt.legend()
    plt.title("Simulated AR(2) Process with Regime Shifts")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()



