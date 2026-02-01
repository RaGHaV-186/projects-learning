import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

data = load_wine()

X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled.shape)

k_values = [2,3,4,5,6]

silhouette_scores = []
db_scores = []

for k in k_values:

    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled,labels)
    db = davies_bouldin_score(X_scaled,labels)

    silhouette_scores.append(sil)
    db_scores.append(db)

    print(f"K={k} | Silhouette: {sil:.3f} (High=Good) | Davies-Bouldin: {db:.3f} (Low=Good)")


fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Silhouette Score (Higher is better)', color=color)
ax1.plot(k_values, silhouette_scores, marker='o', color=color, linewidth=3)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for Davies-Bouldin
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Davies-Bouldin Index (Lower is better)', color=color)
ax2.plot(k_values, db_scores, marker='s', linestyle='--', color=color, linewidth=3)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Validation Duel: Silhouette vs. Davies-Bouldin')
plt.grid(True, alpha=0.3)
plt.show(block=True)