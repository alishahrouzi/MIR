# import pandas as pd
# data1 = pd.read_csv("D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\Merged.csv")
# data = data1.drop(['no','name','genre'],axis=1) #18
# #%%  normalizasyon
# import numpy as np
# from sklearn.preprocessing import  StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(np.array(data, dtype = float))
# #%%
# dat = (X-np.min(X))/(np.max(X)-np.min(X))

# #%%Choosing the Number of Components in a Principal Component Analysis

# from sklearn.decomposition import PCA    
# import matplotlib.pyplot as plt    
# pca = PCA().fit(dat)
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title(' Dataset Explained Variance')
# # plt.show()

# #%%  silhouette_score
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# result = []
# n_clusters=3
# for n_clusters in list (range(3,25)):
#    clusterer = KMeans (n_clusters=n_clusters, init = 'k-means++').fit(X)
#    preds = clusterer.predict(X)
#    centers = clusterer.cluster_centers_
#    result.append(silhouette_score(X, preds, sample_size = 26))

# import matplotlib.pyplot as plt 
# plt.plot(range(3,25), result, 'bx-')
# plt.xlabel('number of clusters')
# plt.ylabel('result')

# # plt.show()      
# #%% 3 boyutlu görselleştirmek için n_components değerini 3 seçtik
# from sklearn.decomposition import PCA
# pca = PCA(n_components=5)
# principalComponents = pca.fit_transform(X)
# #%%
# from sklearn.cluster import KMeans
# n_clusters=5

# clusterer = KMeans (n_clusters=n_clusters, init = 'k-means++').fit(principalComponents)
# preds = clusterer.predict(principalComponents)
# centers = clusterer.cluster_centers_

  
# #%% plt for kmeans
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(8,6))
# ax = Axes3D(fig, elev=-150, azim=110)
# ax.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2], c=preds,cmap=plt.cm.Set1, edgecolor='k')
# #ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], c=aa,cmap=plt.cm.Set1, edgecolor='k')

# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# # plt.show()
# #%%  Hieararchial Clustering
# from sklearn.cluster import AgglomerativeClustering
# import numpy as np
# clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=2.4,compute_full_tree=True).fit(principalComponents)
# #  fit(data) için distance_threshold=3000,   fit(X) için distance_threshold=22,  fit(dat) için distance_threshold=2
# clusters_Sayi=clustering.n_clusters_
# labels=clustering.labels_
# #bb0=list(aa).count(0)
 
# #%%   plt for Hieararchial Clustering
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(8,6))
# ax = Axes3D(fig, elev=-150, azim=110)
# ax.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2], c=labels,cmap=plt.cm.Set1, edgecolor='k')
# #ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], c=aa,cmap=plt.cm.Set1, edgecolor='k')
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# # plt.show()


#------------------------------------------------


import pandas as pd 
import numpy as np 
import random as rd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# data = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\clustering.csv')
data = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\Merged.csv')
data.head()

X = data[['spectral_centroids_mean',"spectral_rolloff_mean"]]
# Y = data[['spectral_centroids_min',
#           'spectral_centroids_mean',
#           'spectral_bandwidth_max',
#           'spectral_bandwidth_mean',
#           'spectral_rolloff_max',
#           'spectral_rolloff_min',
#           'spectral_rolloff_mean',
#           'rms_mean',
#           'mfcc_1_mean',
#           'harmonic_max',
#           'harmonic_min',
#           'harmonic_std',
#           'percussive_max',
#           'percussive_min']]
# plt.scatter(Y['spectral_centroids_min'],
#             Y['spectral_centroids_mean'],
#             ['spectral_bandwidth_max'],
#             Y['spectral_bandwidth_mean'],
#             Y['spectral_rolloff_max'],
#             Y['spectral_rolloff_min'],
#             Y['spectral_rolloff_mean'],
#             Y['rms_mean'],
#             Y['mfcc_1_mean'],
#             Y['harmonic_max'],
#             Y['harmonic_min'],
#             Y['harmonic_std'],
#             Y['percussive_max'],
#             Y['percussive_min'])
plt.scatter(X['spectral_centroids_mean'],X["spectral_rolloff_mean"], c='black')
plt.xlabel('spectral_centroids_mean')
plt.ylabel('spectral_rolloff_mean')
plt.show()

k = 15

Centroids = (X.sample(n=k))
# Centroids = (Y.sample(n=k))
plt.scatter(X['spectral_centroids_mean'],X["spectral_rolloff_mean"], c='black')
plt.scatter(Centroids['spectral_centroids_mean'],Centroids['spectral_rolloff_mean'],c='red')
plt.xlabel('spectral_centroids_mean')
plt.ylabel('spectral_rolloff_mean')
plt.show()


diff = 1 
j= 0

# while (diff != 0):
#    YD = Y
#    i = 1
#    for index1, row_c in Centroids.iterrows():
#       ED = []
#       for index2 , row_d in YD.iterrows():
#          d1 = (row_c['spectral_centroids_mean'] - row_d['spectral_centroids_mean'])**2
#          d2 = (row_c['spectral_rolloff_mean'] - row_d['spectral_rolloff_mean'])**2
#          d = np.sqrt(d1 + d2)
#          ED.append(d) 
#       Y[i] = ED
#       i = i + 1
#    C =[]
#    for index,row in Y.iterrows():
#       min_dist = row[1]
#       pos = 1 
#       for i in range(k):
#          if row[i+1] < min_dist:
#             min_dist = row[i+1]
#             pos = i + 1
#       C.append(pos)
#    Y['Cluster'] = C
#    Centroids_new = Y.groupby(['Cluster']).mean()[['spectral_rolloff_mean', 'spectral_centroids_mean']]
#    if j == 0 :
#       diff = 1 
#       j = j +1 
#    else :
#       diff = (Centroids_new['spectral_rolloff_mean'] - Centroids_new['spectral_centroids_mean']).sum() + (Centroids_new['spectral_centroids_mean'] - Centroids_new['spectral_rolloff_mean']).sum()
#       print(diff.sum())
#    Centroids = Y.groupby(['Cluster']).mean()[['spectral_rolloff_mean', 'spectral_centroids_mean']]





while (diff != 0):
   XD = X
   i = 1
   for index1, row_c in Centroids.iterrows():
      ED = []
      for index2 , row_d in XD.iterrows():
         d1 = (row_c['spectral_centroids_mean'] - row_d['spectral_centroids_mean'])**2
         d2 = (row_c['spectral_rolloff_mean'] - row_d['spectral_rolloff_mean'])**2
         d = np.sqrt(d1 + d2)
         ED.append(d) 
      X[i] = ED
      i = i + 1
   C =[]
   for index,row in X.iterrows():
      min_dist = row[1]
      pos = 1 
      for i in range(k):
         if row[i+1] < min_dist:
            min_dist = row[i+1]
            pos = i + 1
      C.append(pos)
   X['Cluster'] = C
   Centroids_new = X.groupby(['Cluster']).mean()[['spectral_rolloff_mean', 'spectral_centroids_mean']]
   if j == 0 :
      diff = 1 
      j = j +1 
   else :
      diff = (Centroids_new['spectral_rolloff_mean'] - Centroids_new['spectral_centroids_mean']).sum() + (Centroids_new['spectral_centroids_mean'] - Centroids_new['spectral_rolloff_mean']).sum()
      print(diff.sum())
   Centroids = X.groupby(['Cluster']).mean()[['spectral_rolloff_mean', 'spectral_centroids_mean']]    


color = ['blue', 'green' , 'red' , 'yellow' , 'brown','Orange','Purple','Cyan','Magenta','Teal','Pink','Lime','Indigo','Maroon','Olive','Coral']
for k in range(k):
   data = X[X['Cluster']==k+1]
   plt.scatter(data['spectral_centroids_mean'],data['spectral_rolloff_mean'],c =color[k])
plt.scatter(Centroids['spectral_centroids_mean'],Centroids['spectral_rolloff_mean'],c = 'red')
plt.xlabel('spectral_centroids_mean')
plt.ylabel('spectral_rolloff_mean')
plt.show()



# print(data.describe())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_m = data.drop(['name','no','genre'], axis = 1)
data_scaled = scaler.fit_transform(data_m)
# print(pd.DataFrame(data_scaled).describe())

kmeans = KMeans(n_clusters=2 , init='k-means++')
kmeans.fit(data_scaled)

# print(kmeans.inertia_)


SSE = []
for cluster in range(1,20):
   kmeans = KMeans(n_clusters=cluster , init='k-means++')
   kmeans.fit(data_scaled)
   SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker = 'o')
plt.xlabel('Num of Cluster')
plt.ylabel('Inertia')
plt.title(kmeans.inertia_)
# plt.show()