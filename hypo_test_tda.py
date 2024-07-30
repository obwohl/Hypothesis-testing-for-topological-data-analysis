import numpy as np
import umap
from ripser import ripser
from scipy.stats import wasserstein_distance
from scipy import stats
from itertools import combinations
from sklearn.datasets import load_iris, fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import hashlib
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly.subplots import make_subplots



def create_point_clouds_from_dataset(sample_size=500):
    iris = load_iris()
    cali = fetch_california_housing()
    
    def process_dataset(dataset, size, name):
        data = dataset.data
        feature_names = dataset.feature_names
        
        if name == 'iris' and data.shape[1] != len(feature_names):
            print(f"Warning: Mismatch in iris dataset. Data has {data.shape[1]} columns, but {len(feature_names)} feature names.")
            if data.shape[1] == len(feature_names) + 1:
                print("Assuming last column is 'target'. Removing it from the data.")
                data = data[:, :-1]
            else:
                print(f"Using generic column names for {name} dataset.")
                feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        df = pd.DataFrame(data, columns=feature_names)
        
        actual_size = min(len(df), size)
        if actual_size < size:
            print(f"Warning: {name} dataset has only {actual_size} samples. Using all available samples.")
        
        df = df.sample(n=actual_size, random_state=42)
        return df.values, df.columns.tolist()
    
    iris_data, iris_features = process_dataset(iris, sample_size, 'iris')
    cali_data, cali_features = process_dataset(cali, sample_size, 'california')
    
    print(f"Iris dataset shape: {iris_data.shape}")
    print(f"California Housing dataset shape: {cali_data.shape}")
    
    print("\nBasic statistics for Iris dataset:")
    print(pd.DataFrame(iris_data, columns=iris_features).describe())
    print("\nBasic statistics for California Housing dataset:")
    print(pd.DataFrame(cali_data, columns=cali_features).describe())
    
    return iris_data, cali_data


def create_point_clouds_from_dataset(sample_size=500):
    def generate_torus(n_points, R=1, r=0.3):
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        return np.vstack((x, y, z)).T

    def generate_torus_mug(n_points, R=1, r=0.3):
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        mug_handle_theta = np.random.uniform(0, 2*np.pi, n_points // 5)
        mug_handle_x = R + r * np.cos(mug_handle_theta)
        mug_handle_y = r * np.sin(mug_handle_theta)
        mug_handle_z = np.random.uniform(-r, r, n_points // 5)
        x = np.concatenate((x, mug_handle_x))
        y = np.concatenate((y, mug_handle_y))
        z = np.concatenate((z, mug_handle_z))
        return np.vstack((x, y, z)).T
    
    # Sample points for torus and torus-mug
    torus_points = generate_torus(sample_size)
    torus_mug_points = generate_torus_mug(sample_size)

    # Convert to DataFrame for consistency with original function
    torus_df = pd.DataFrame(torus_points, columns=['x', 'y', 'z'])
    torus_mug_df = pd.DataFrame(torus_mug_points, columns=['x', 'y', 'z'])

    # Print basic statistics
    print(f"Torus dataset shape: {torus_df.shape}")
    print(f"Torus Mug dataset shape: {torus_mug_df.shape}")
    
    print("\nBasic statistics for Torus dataset:")
    print(torus_df.describe())
    print("\nBasic statistics for Torus Mug dataset:")
    print(torus_mug_df.describe())
    
    return torus_df.values, torus_mug_df.values

def plot_3d_point_cloud(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    plt.show()

# generate points
torus_points, torus_mug_points = create_point_clouds_from_dataset(sample_size=5000)

# plot it
plot_3d_point_cloud(torus_points, 'Torus')
plot_3d_point_cloud(torus_mug_points, 'Torus Mug')


def create_point_clouds_from_dataset(sample_size=500):
    def generate_torus(n_points, R=1, r=0.3):
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        return np.vstack((x, y, z)).T

    def generate_sphere(n_points, radius=1):
        phi = np.random.uniform(0, np.pi, n_points)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.vstack((x, y, z)).T
    
    # Sample points for torus and sphere
    torus_points = generate_torus(sample_size)
    sphere_points = generate_sphere(sample_size)

    # Convert to DataFrame for consistency with original function
    torus_df = pd.DataFrame(torus_points, columns=['x', 'y', 'z'])
    sphere_df = pd.DataFrame(sphere_points, columns=['x', 'y', 'z'])

    # Print basic statistics
    print(f"Torus dataset shape: {torus_df.shape}")
    print(f"Sphere dataset shape: {sphere_df.shape}")
    
    print("\nBasic statistics for Torus dataset:")
    print(torus_df.describe())
    print("\nBasic statistics for Sphere dataset:")
    print(sphere_df.describe())
    
    return torus_df.values, sphere_df.values

def plot_3d_point_cloud(points, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2)
    )])
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    fig.show()

# generate points
torus_points, sphere_points = create_point_clouds_from_dataset(sample_size=500)

# plot them
plot_3d_point_cloud(torus_points, 'Torus')
plot_3d_point_cloud(sphere_points, 'Sphere')

def visualize_data(A, B, title_A, title_B):
    pca_A = PCA(n_components=2)
    pca_B = PCA(n_components=2)
    
    A_vis = pca_A.fit_transform(A)
    B_vis = pca_B.fit_transform(B)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(A_vis[:, 0], A_vis[:, 1], alpha=0.5)
    plt.title(f"{title_A} (PCA)")
    plt.subplot(122)
    plt.scatter(B_vis[:, 0], B_vis[:, 1], alpha=0.5)
    plt.title(f"{title_B} (PCA)")
    plt.tight_layout()
    plt.show()

def check_and_adjust_dimensions(A, B):
    dim_A, dim_B = A.shape[1], B.shape[1]
    if dim_A != dim_B:
        target_dim = min(dim_A, dim_B)
        print(f"Reducing dimensions to {target_dim} with UMAP")
        reducer = umap.UMAP(n_components=target_dim)
        A = reducer.fit_transform(A) if dim_A > target_dim else A
        B = reducer.fit_transform(B) if dim_B > target_dim else B
    print("UMAP reduction completed")
    
    print("\nSample data after dimension reduction:")
    print("A (first 5 rows):")
    print(A[:5])
    print("B (first 5 rows):")
    print(B[:5])
    
    return A, B

def compute_persistence_diagram(point_cloud):
    diagrams = ripser(point_cloud)['dgms']
    return [[(b, d) for b, d in diag if b != d and np.isfinite(b) and np.isfinite(d)] for diag in diagrams]




def hash_data(data):
    """
    Nimmt eine Liste von Punkten und gibt einen SHA-256 Hash der verketteten Daten zurück.
    """
    concatenated_data = np.array(data).flatten().tobytes()
    return hashlib.sha256(concatenated_data).hexdigest()





# check mps
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS nicht verfügbar, weil die aktuelle PyTorch-Installation nicht für MPS gebaut wurde.")
    else:
        print("MPS nicht verfügbar, obwohl PyTorch für MPS gebaut wurde.")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

def wilson_score_interval(count, nobs, alpha=0.05):
    n = nobs
    p = count / n
    z = stats.norm.ppf(1 - alpha / 2)
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    
    return (center - spread, center + spread)

def compute_test_statistic_gpu(diagrams_A, diagrams_B, metric='F_{2,2}'):
    if metric == 'F_{2,2}':
        dist_A = torch.cdist(diagrams_A, diagrams_A)
        dist_B = torch.cdist(diagrams_B, diagrams_B)
        return torch.mean(dist_A**2) + torch.mean(dist_B**2)
    elif metric == 'F_{1,1}':
        dist_A = torch.cdist(diagrams_A, diagrams_A, p=1)
        dist_B = torch.cdist(diagrams_B, diagrams_B, p=1)
        return torch.mean(dist_A) + torch.mean(dist_B)
    else:
        raise ValueError(f"Unbekannte Metrik: {metric}")

def compute_batch_permutations_gpu(all_diagrams, n_A, metric, batch_size, observed_statistic):
    n_total = all_diagrams.shape[0]
    
    # Generieren Sie Zufallsindizes für die Permutationen
    random_indices = torch.randint(0, n_total, (batch_size, n_total), device=device)
    
    # Extrahieren Sie permutierte Diagramme
    permuted_A = all_diagrams[random_indices[:, :n_A]]
    permuted_B = all_diagrams[random_indices[:, n_A:]]
    
    # Berechnen Sie die Teststatistik für jede Permutation
    permuted_stats = compute_test_statistic_gpu(permuted_A, permuted_B, metric)
    
    # Vergleichen Sie mit der beobachteten Statistik
    results = permuted_stats >= observed_statistic
    
    return results.cpu().numpy().flatten()



def gpu_optimized_permutation_test(diagrams_A, diagrams_B, metric='F_{2,2}', num_permutations=10000, alpha=0.05, early_stop_threshold=1000, batch_size=32):
    combined_A = torch.tensor(combine_persistence_diagrams(diagrams_A), device=device, dtype=torch.float32)
    combined_B = torch.tensor(combine_persistence_diagrams(diagrams_B), device=device, dtype=torch.float32)
    
    all_diagrams = torch.cat([combined_A, combined_B], dim=0)
    observed_statistic = compute_test_statistic_gpu(combined_A, combined_B, metric).item()
    print(f"Observed Test Statistic: {observed_statistic}")
    
    n_A = combined_A.shape[0]
    
    permuted_statistics = []
    count_exceeding = 0
    total_permutations = 0
    p_value_progression = []
    
    start_time = time.time()
    
    try:
        while total_permutations < num_permutations:
            current_batch_size = min(batch_size, num_permutations - total_permutations)
            batch_results = compute_batch_permutations_gpu(all_diagrams, n_A, metric, current_batch_size, observed_statistic)
            
            if batch_results.size == 0:
                print(f"Warning: Empty batch result at permutation {total_permutations}")
                continue
            
            permuted_statistics.extend(batch_results)
            count_exceeding += np.sum(batch_results)
            total_permutations += current_batch_size
            
            current_p_value = count_exceeding / total_permutations
            p_value_progression.append((total_permutations, current_p_value))
            
            ci_low, ci_high = wilson_score_interval(count_exceeding, total_permutations, alpha=alpha)
            
            elapsed_time = time.time() - start_time
            print(f"Permutation {total_permutations}: p-value = {current_p_value:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}], Time: {elapsed_time:.2f}s")
            
            if total_permutations >= early_stop_threshold and (ci_low > alpha or ci_high < alpha):
                print(f"Early stopping after {total_permutations} permutations: p-value is with 95% confidence {'above' if ci_low > alpha else 'below'} {alpha}")
                break
    
    except Exception as e:
        print(f"Error during computation: {e}")
        print(f"Current number of permutations: {total_permutations}")
    
    final_p_value = count_exceeding / total_permutations if total_permutations > 0 else None
    return final_p_value, observed_statistic, permuted_statistics, total_permutations, p_value_progression


def create_result_dataframe(p_value, observed_statistic, num_permutations, metric, alpha=0.01):
    confidence_levels = {0.01: 0.99, 0.05: 0.95, 0.1: 0.90}
    confidence = next((conf for p, conf in confidence_levels.items() if p_value <= p), 1 - p_value)
    
    interpretation = (
        'Strong evidence against the null hypothesis' if p_value <= 0.01 else
        'Moderate evidence against the null hypothesis' if p_value <= 0.05 else
        'Weak evidence against the null hypothesis' if p_value <= 0.1 else
        'No significant evidence against the null hypothesis'
    )
    
    return pd.DataFrame({
        'Metric': [metric],
        'P-Value': [p_value],
        'Observed Statistic': [observed_statistic],
        'Number of Permutations': [num_permutations],
        'Significance Level': [alpha],
        'Null Hypothesis': ['The two sets of persistence diagrams come from the same distribution'],
        'Null Hypothesis Rejected': [p_value <= alpha],
        'P-Value Percentile': [(1 - p_value) * 100],
        'Interpretation': [interpretation],
        'Confidence': [confidence]
    })

def plot_persistence_diagrams(diagrams, title='Persistence Diagram'):
    plt.figure(figsize=(10, 10))
    for i, diag in enumerate(diagrams):
        if len(diag) > 0:
            birth, death = zip(*diag)
            plt.scatter(birth, death, label=f'H{i}', alpha=0.6)
    
    max_value = max(max(d for diag in diagrams for _, d in diag), 1)
    plt.plot([0, max_value], [0, max_value], 'k--', label='Diagonal')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def combine_persistence_diagrams(diagrams):
    combined = []
    for dim, diagram in enumerate(diagrams):
        for point in diagram:
            combined.append([dim] + list(point))
    return combined

def wasserstein_distance(diag1, diag2, p=2):
    # Convert generators to lists if necessary
    diag1 = list(diag1) if hasattr(diag1, '__iter__') else [diag1]
    diag2 = list(diag2) if hasattr(diag2, '__iter__') else [diag2]
    
    # Ensure diag1 and diag2 are numpy arrays
    diag1 = np.array(diag1)
    diag2 = np.array(diag2)
    
    # If diagrams are single points, compute distance directly
    if diag1.shape == (3,) and diag2.shape == (3,):
        return np.linalg.norm(diag1 - diag2)
    
    # Check if the diagrams are empty
    if len(diag1) == 0 and len(diag2) == 0:
        return 0.0
    elif len(diag1) == 0:
        return np.sum(np.abs(diag2[:, 1] - diag2[:, 0])**p)**(1/p)
    elif len(diag2) == 0:
        return np.sum(np.abs(diag1[:, 1] - diag1[:, 0])**p)**(1/p)
    
    # Ensure diagrams are 2-dimensional
    if diag1.ndim == 1:
        diag1 = diag1.reshape(1, -1)
    if diag2.ndim == 1:
        diag2 = diag2.reshape(1, -1)
    
    # Separate the dimension information and the birth-death coordinates
    dim1, points1 = diag1[:, 0], diag1[:, 1:]
    dim2, points2 = diag2[:, 0], diag2[:, 1:]
    
    # Compute pairwise distances, considering dimension information
    M = cdist(points1, points2, metric='euclidean')
    
    # Adjust distances for points in different dimensions
    dim_diff = np.abs(dim1[:, np.newaxis] - dim2[np.newaxis, :])
    M += dim_diff * np.max(M)  # Penalize matching across dimensions
    
    # Add diagonal points
    diag1_to_diag = np.sum(np.abs(points1 - np.array([(p[0]+p[1])/2, (p[0]+p[1])/2]) for p in points1) ** p, axis=1) ** (1/p)
    diag2_to_diag = np.sum(np.abs(points2 - np.array([(p[0]+p[1])/2, (p[0]+p[1])/2]) for p in points2) ** p, axis=1) ** (1/p)
    
    M = np.pad(M, ((0, len(diag2)), (0, len(diag1))), mode='constant', constant_values=0)
    M[-len(diag2):, :len(diag1)] = diag1_to_diag[:, np.newaxis]
    M[:len(diag1), -len(diag2):] = diag2_to_diag[np.newaxis, :]
    
    # Solve the optimal transport problem
    row_ind, col_ind = linear_sum_assignment(M ** p)
    
    return (M[row_ind, col_ind].sum()) ** (1/p)
def compute_metric(diagrams, metric='F_{2,2}'):
    n = len(diagrams)
    if n <= 1:
        return 0
    
    total_distance = 0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                distance = wasserstein_distance(diagrams[i], diagrams[j], p=2)
                if metric == 'F_{2,2}':
                    total_distance += distance ** 2
                elif metric == 'F_{1,1}':
                    total_distance += distance
                count += 1
            except Exception as e:
                print(f"Error computing distance between diagram {i} and {j}: {e}")
                print(f"Diagram {i} shape: {np.array(diagrams[i]).shape}")
                print(f"Diagram {j} shape: {np.array(diagrams[j]).shape}")
    
    if count == 0:
        return 0
    
    if metric == 'F_{2,2}':
        return total_distance / count
    elif metric == 'F_{1,1}':
        return total_distance / count
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compute_test_statistic(combined_A, combined_B, metric='F_{2,2}'):
    return compute_metric(combined_A, metric) + compute_metric(combined_B, metric)


import matplotlib.pyplot as plt

def plot_p_value_progression(p_value_progressions, metrics, alpha=0.05):
    plt.figure(figsize=(12, 6))
    for progression, metric in zip(p_value_progressions, metrics):
        permutations, p_values = zip(*progression)
        plt.plot(permutations, p_values, label=f'Metric: {metric}')
    
    plt.axhline(y=alpha, color='r', linestyle='--', label='Significance Level')
    plt.xlabel('Number of Permutations')
    plt.ylabel('p-value')
    plt.title('p-value Progression During Permutation Test')
    plt.legend()
    plt.yscale('log')  # Logarithmische Skala für bessere Sichtbarkeit
    plt.grid(True)
    plt.show()




def print_final_results(p_value, observed_statistic, total_permutations, metric, alpha):
    print("\n--- Finale Ergebnisse ---")
    print(f"Metrik: {metric}")
    print(f"Beobachtete Teststatistik: {observed_statistic:.4f}")
    print(f"p-Wert: {p_value:.4f}")
    ci_low, ci_high = wilson_score_interval(int(p_value * total_permutations), total_permutations, alpha=alpha)
    print(f"95% Konfidenzintervall für p-Wert: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Anzahl der Permutationen: {total_permutations}")
    print(f"Signifikanzniveau: {alpha}")
    print(f"Nullhypothese {'verworfen' if p_value <= alpha else 'nicht verworfen'} bei α = {alpha}")
    if p_value <= alpha:
        print("Es gibt signifikante Unterschiede zwischen den Persistenzdiagrammen der beiden Datensätze.")
    else:
        print("Es gibt keine signifikanten Unterschiede zwischen den Persistenzdiagrammen der beiden Datensätze.")





def plot_p_value_progression(p_value_progression, metric, alpha=0.05):
    permutations, p_values = zip(*p_value_progression)
    plt.figure(figsize=(12, 6))
    plt.plot(permutations, p_values, label=f'Metric: {metric}')
    
    plt.axhline(y=alpha, color='r', linestyle='--', label='Significance Level')
    plt.xlabel('Number of Permutations')
    plt.ylabel('p-value')
    plt.yscale('log') 
    plt.title(f'p-value Progression During Permutation Test for {metric}')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_analysis(num_permutations=10000, sample_size=500, alpha=0.05):

    A, B = create_point_clouds_from_dataset(sample_size)
    print(f"Ursprüngliche Form A: {A.shape}    Form B: {B.shape}")
    
    visualize_data(A, B, "Torus", "Sphere")
    
    A, B = check_and_adjust_dimensions(A, B)
    print(f"Angepasste Form A: {A.shape}    Form B: {A.shape}")
    
    visualize_data(A, B, "Torus (Nach UMAP)", "Sphere (Nach UMAP)")
    

    print("Berechne Persistenzdiagramme...")
    diagrams_A = compute_persistence_diagram(A)
    diagrams_B = compute_persistence_diagram(B)

    plot_persistence_diagrams(diagrams_A, title='Persistenzdiagramm für Torus')
    plot_persistence_diagrams(diagrams_B, title='Persistenzdiagramm für Sphere')
    

    results = []
    p_value_progressions = []
    for metric in ['F_{2,2}', 'F_{1,1}']:
        print(f"\nFühre Permutationstest mit Metrik {metric} durch...")
        p_value, observed_statistic, permuted_statistics, total_permutations, p_value_progression = gpu_optimized_permutation_test(
            diagrams_A, diagrams_B, metric=metric, 
            num_permutations=num_permutations, alpha=alpha, batch_size=32
        )
        print_final_results(p_value, observed_statistic, total_permutations, metric, alpha)
        results.append([metric, p_value, observed_statistic, total_permutations])
        
        p_value_progressions.append(p_value_progression)
        plot_p_value_progression(p_value_progression, metric, alpha)
    

    final_results = pd.DataFrame(results, columns=['Metrik', 'p-Wert', 'Beobachtete Teststatistik', 'Anzahl Permutationen'])
    print("\nZusammenfassung aller Tests:")
    print(final_results.to_string(index=False))
    final_results.to_csv('permutation_test_results.csv', index=False)
    
    return final_results, diagrams_A, diagrams_B

if __name__ == "__main__":
    results, diag_a, diag_b = run_analysis(num_permutations=1000, sample_size=2000)




