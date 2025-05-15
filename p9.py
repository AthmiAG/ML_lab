import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mode

df = pd.read_csv("Breast Cancer Wisconsin.csv")
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

if 'diagnosis' in df.columns:
    diagnosis = df['diagnosis'].map({'M': 1, 'B': 0})
    df = df.drop(columns=['diagnosis'])
else:
    diagnosis = None

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

vis_df = pd.DataFrame({
    'PCA1': pca_data[:, 0],
    'PCA2': pca_data[:, 1],
    'Cluster': labels
})

if diagnosis is not None:
    vis_df['Actual'] = diagnosis
    cluster_map = {}
    for cluster in [0, 1]:
        majority = mode(diagnosis[labels == cluster], keepdims=True).mode[0]
        cluster_map[cluster] = 'Predicted Malignant' if majority == 1 else 'Predicted Benign'
    vis_df['Cluster_Label'] = vis_df['Cluster'].map(cluster_map)
    vis_df['Actual_Label'] = vis_df['Actual'].map({1: 'Actual Malignant', 0: 'Actual Benign'})

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=vis_df,
        x='PCA1', y='PCA2',
        hue='Cluster_Label',
        palette={'Predicted Malignant': 'red', 'Predicted Benign': 'blue'},
        s=100,
        alpha=0.6,
        legend='full'
    )
    sns.scatterplot(
        data=vis_df,
        x='PCA1', y='PCA2',
        style='Actual_Label',
        markers={'Actual Malignant': 'X', 'Actual Benign': 'o'},
        color='black',
        s=50,
        alpha=0.5,
        legend='brief'
    )
    plt.title("K-Means Clustering on Breast Cancer Dataset (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
