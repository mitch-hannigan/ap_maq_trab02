import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, homogeneity_score, adjusted_rand_score, completeness_score, normalized_mutual_info_score
from scipy.spatial.distance import pdist
count_clusters = lambda labels:len(set(labels)) - (1 if -1 in labels else 0)
def get_coe(clusterizer):
    print("coesão: ", clusterizer.inertia_)
def get_sep(clusterizer):
    print("separação: ", np.sum(pdist(clusterizer.cluster_centers_, metric='euclidean')))
def get_silhouette_score(atributos, clusterizer):
    masc = clusterizer.labels_ != -1
    print("coeficiente de Silhueta médio:", silhouette_score(atributos[masc], clusterizer.labels_[masc]))
def get_scores(classes, clusterizer):
    masc = clusterizer.labels_ != -1
    print("homogeneidade:", homogeneity_score(classes[masc], clusterizer.labels_[masc]))
    print("índice randômico:", adjusted_rand_score(classes[masc], clusterizer.labels_[masc]))
    print("completude:", completeness_score(classes[masc], clusterizer.labels_[masc]))
    print("entropia:", normalized_mutual_info_score(classes[masc], clusterizer.labels_[masc]))
data, meta = arff.loadarff('Raisin_Dataset.arff')
df = pd.DataFrame(data)
atributos = df.iloc[:,:-1]
classes = df.iloc[:,-1]
def calibrate_kmeans():
    print("kmeans: ")
    for c in range(2, 10):
        for i in range(10, 300, 20):
            km = KMeans(c, max_iter=i)
            print("num clusters: ", c)
            print("max_iters:", i)
            km=km.fit(atributos)
            get_coe(km)
            get_sep(km)
            get_silhouette_score(atributos, km)
            get_scores(classes, km)
def calibrate_dbscan():
    print("dbscan: ")
    for e in np.arange(1, 1000, 5):
        for ms in range(2, 20):
            db = DBSCAN(eps=e, min_samples=ms)
            db.fit(atributos)
            if np.unique(db.labels_[db.labels_!=-1]).size>=2: 
                print("eps:", e)
                print("min_samples", ms)
                print("samples descartados: ", np.count_nonzero(db.labels_==-1))
                #get_coe(db)
                #get_sep(db)
                get_silhouette_score(atributos, db)
                get_scores(classes, db)
#calibrate_kmeans()
calibrate_dbscan()