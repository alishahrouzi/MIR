import pandas as pd
import plotly.express as px
from sklearn.metrics import silhouette_score

# Load the clusters output
# clusters_output = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\clusters_output_15Clusterd.csv')
# clusters_output = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\clusters_output_5Clusterd.csv')
clusters_output = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Sample\\clusters_sample_5 - labeled.csv')
# Extract features and cluster labels
X = clusters_output.iloc[:, 3:-1]  # Exclude the first column (sample names) and last column (cluster labels)
cluster_labels = clusters_output['Cluster']

# Compute silhouette score
silhouette_avg = silhouette_score(X, cluster_labels)
print("Silhouette Score:", silhouette_avg)

# Plot clusters
fig = px.scatter_matrix(X, dimensions=X.columns, color=cluster_labels,
                        title="Clusters Visualization (Hover to see sample names)",
                        labels={'color': 'Cluster'})
fig.update_traces(marker=dict(size=5),
                  diagonal_visible=False)
fig.update_layout(width=1500, height=1000)
fig.show(renderer="browser")  # Display the plot in a web browser
