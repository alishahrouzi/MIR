import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time
import datetime

# Function to display time in HH:MM:SS format
def display_time(seconds):
    return str(datetime.timedelta(seconds=seconds))

def standardize_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Importing the data
print("Importing data...")
# data = pd.read_csv('Merged.csv')
combination = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0]
data = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\clusters_output_15Clusterd.csv')
X = data
X = data.drop(['name','Cluster','label',],axis=1)
removed_columns = []
for i in range(len(combination)):
    if combination[i] == 0:
        removed_columns.append(X.columns[i])
        
X = X.drop(removed_columns,axis=1)
# X = data.drop(['no','genre','name'],axis=1)
scaler = MinMaxScaler()
standard = StandardScaler()
X = scaler.fit_transform(X)
X = standard.fit_transform(X)


# print("Extracting features...")
# l0 = list(range(3, 7))
# l1 = list(range(19, 23))
# l2 = list(range(31, 67))
# l3 = list(range(79, 167))
# l = zip(l0,l1,l2,l3)
# temp = []
# for i in range(len(l0)):
#     temp.append(l0[i])
 
# for i in range(len(l1)):
#     temp.append(l1[i])   
    
# for i in range(len(l2)):
#     temp.append(l2[i])
    
# for i in range(len(l3)):
#     temp.append(l3[i])
# X = data.iloc[:, (t for t in temp)]





result = []
# for n_clusters in list (range(3,100)):
#    clusterer = KMeans (n_clusters=n_clusters, init = 'k-means++').fit(X)
#    preds = clusterer.predict(X)
#    centers = clusterer.cluster_centers_
#    result.append(silhouette_score(X, preds, sample_size = 300))
# print(result)

# import matplotlib.pyplot as plt 
# plt.plot(range(3,100), result, 'bx-')
# plt.xlabel('number of clusters')
# plt.ylabel('result')

# plt.show()  


# Loop over different numbers of clusters
for num_clusters in range(2, 11):  # Range of clusters to explore
    print(f"Trying {num_clusters} clusters...")
    
    # Clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = hierarchical.fit_predict(X)
    
    # Compute silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("Silhouette Score:", silhouette_avg)
    best_score = 0
    best_num_clusters = []
    # Update best score and number of clusters if current configuration is better
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        # best_num_clusters = num_clusters
        best_num_clusters.append(num_clusters)

print(f"Best number of clusters: {best_num_clusters}")
print("Best silhouette score:", best_score)

# Final clustering with the best number of clusters
print("Applying hierarchical clustering algorithm with the best number of clusters...")
start_time = time.time()
hierarchical = AgglomerativeClustering(n_clusters=max(best_num_clusters))
cluster_labels = hierarchical.fit_predict(X)
end_time = time.time()
print("Clustering completed in", display_time(end_time - start_time))

# # Saving clusters to a file
print("Saving clusters to file...")
data['Cluster'] = best_num_clusters
data.to_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\clusters15_output_1.csv', index=False)

# Displaying clusters
print("Displaying clusters:")
for cluster_num in range(max(best_num_clusters)):
    print("Cluster", cluster_num)
    cluster_data = data[data['Cluster'] == cluster_num]
    print(cluster_data)
    print("\n")
