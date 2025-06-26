# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:18:55 2025

@author: eggra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, anderson, normaltest
from sklearn.decomposition import PCA

# Load simulation results
results = np.load("karmic_vector_stochastic_results.npy")  # Shape: (15000, 3, 7)

# ===== 1. Data Preparation =====
# Reshape results into pandas DataFrame
metrics = ['max_I', 't_max', 'area_I', 'n_peaks', 'lyapunov', 'apen', 'corr_dim']
components = ['avidya', 'raga', 'dvesha']

# Create multi-index DataFrame
sim_data = pd.DataFrame(
    results.reshape(-1, 7),
    columns=metrics
)
sim_data['component'] = np.repeat(components, len(results))
sim_data['simulation'] = np.tile(np.arange(len(results)), 3)

# ===== 2. Basic Statistical Analysis =====
def basic_stats(data):
    stats = data.groupby('component').agg({
        'max_I': ['mean', 'std', lambda x: kstest(x, 'norm').statistic],
        'lyapunov': ['mean', lambda x: (x > 0).mean()],  # Chaos probability
        'apen': ['mean', lambda x: (x > 0.5).mean()],    # Complexity probability
        'corr_dim': ['mean', 'median']
    })
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return stats

basic_statistics = basic_stats(sim_data)
print("Basic Statistics:")
print(basic_statistics.to_markdown())

# ===== 3. Chaos Detection =====
def detect_chaos(df):
    chaos_criteria = (
        (df['lyapunov'] > 0) & 
        (df['apen'] > 0.5) & 
        (df['corr_dim'] > 1.5))
    return df.groupby('component')[chaos_criteria].mean()

chaos_probability = detect_chaos(sim_data)
print("\nChaos Probability per Component:")
print(chaos_probability.to_markdown())

# ===== 4. Cross-Component Analysis =====
cross_corr = sim_data.pivot(
    index='simulation', 
    columns='component', 
    values=['max_I', 'lyapunov']
).corr()

print("\nCross-Component Correlations:")
plt.figure(figsize=(10,8))
sns.heatmap(cross_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix Between Karmic Components")
plt.savefig('cross_component_corr.png')

# ===== 5. Dimensionality Reduction =====
pca = PCA(n_components=2)
pca_results = pca.fit_transform(results.reshape(len(results), -1))

plt.figure(figsize=(10,6))
plt.scatter(pca_results[:,0], pca_results[:,1], alpha=0.5)
plt.xlabel("PC1 (Dynamic Complexity)")
plt.ylabel("PC2 (Manifestation Intensity)")
plt.title("PCA of Karmic Simulation Space")
plt.savefig('pca_analysis.png')

# ===== 6. Advanced Statistical Tests =====
def normality_tests(data):
    tests = {}
    for comp in components:
        comp_data = data[data['component'] == comp]
        tests[comp] = {
            'KS': kstest(comp_data['lyapunov'], 'norm'),
            'Anderson': anderson(comp_data['max_I']),
            'Normaltest': normaltest(comp_data['apen'])
        }
    return tests

normality_results = normality_tests(sim_data)

# ===== 7. Generate Report =====
with open('karmic_analysis_report.md', 'w') as f:
    f.write("# Karmic Vector Model Statistical Analysis\n\n")
    f.write("## Basic Statistics\n")
    f.write(basic_statistics.to_markdown() + "\n\n")
    f.write("## Chaos Probability\n")
    f.write(chaos_probability.to_markdown() + "\n\n")
    f.write("![Cross-Component Correlations](cross_component_corr.png)\n\n")
    f.write("![PCA Analysis](pca_analysis.png)\n\n")
    f.write("## Normality Tests\n```python\n")
    f.write(str(normality_results) + "\n```")

print("Analysis complete. Results saved to:")
print("- karmic_analysis_report.md")
print("- cross_component_corr.png")
print("- pca_analysis.png")