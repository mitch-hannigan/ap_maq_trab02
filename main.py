import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, homogeneity_score, adjusted_rand_score, completeness_score, normalized_mutual_info_score 
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
count_clusters = lambda labels:len(set(labels)) - (1 if -1 in labels else 0)
def calc_coe(X, labels):
    final = 0
    for cluster in set(labels):
        if cluster == -1:
            continue
        cluster_points = X[labels == cluster]
        distances = euclidean_distances(cluster_points)
        cluster_coe = np.sum(distances) / 2
        final += cluster_coe
    return final
def get_coe(clusterizer, km=True, atributos=None):
    final = {"coe":clusterizer.inertia_ if km else calc_coe(atributos, clusterizer.labels_)}
    return final
def calculate_sep(X, labels):
    final = 0
    masc = labels != -1
    X_filtered = X[masc]
    labels_filtered = labels[masc]
    clusters = np.unique(labels_filtered)
    for cluster in clusters:
        cluster_points = X_filtered[labels_filtered == cluster]
        outside_points = X_filtered[labels_filtered != cluster]
        if outside_points.size > 0:
            distance = euclidean_distances(cluster_points, outside_points)
            final += np.sum(distance)/(cluster_points.size*outside_points.size)
    return final
def get_sep(clusterizer, scores, km=True, atributos=None):
    scores["sep"] = np.sum(pdist(clusterizer.cluster_centers_, metric='euclidean')) if km else calculate_sep(atributos, clusterizer.labels_)
    return scores
def get_silhouette_score(atributos, clusterizer, scores):
    masc = clusterizer.labels_ != -1
    scores["s"] = silhouette_score(atributos[masc], clusterizer.labels_[masc])
    return scores
def get_scores(classes, clusterizer, scores):
    masc = clusterizer.labels_ != -1
    scores["h"]=homogeneity_score(classes[masc], clusterizer.labels_[masc])
    scores["i"]=adjusted_rand_score(classes[masc], clusterizer.labels_[masc])
    scores["c"] = completeness_score(classes[masc], clusterizer.labels_[masc])
    scores["e"] = normalized_mutual_info_score(classes[masc], clusterizer.labels_[masc])
    return scores
def get_best(params, scores):
    def calculate_score(scores):
        return scores["h"]+scores["c"]+scores["e"]+(scores["s"]+1)/2+(scores["i"]+1)/2
    #(scores["sep"]-1)*-1+(+scores["coe"]-1)*-1
    best, score = 0, 0
    for i in range(len(params)):
        if(calculate_score(scores[i])>score):
            best=i
            score=calculate_score(scores[i])
    return params[best], scores[best]
def get_min_max(scores):
    min, max={"coe":2, "sep":2},{"coe":-1, "sep":-1}
    for i in scores:
        for k in ["coe", "sep"]:
            if(i[k]<min[k]):min[k]=i[k]
            elif(i[k]>max[k]):max[k]=i[k]
    return min, max
def normalize_scores(scores):
    m1, m2 = get_min_max(scores)
    for i in range(len(scores)):
        for k in ["coe", "sep"]:
            scores[i][k] = (scores[i][k]-m1[k])/(m2[k]-m1[k])
    return scores
data, meta = arff.loadarff('Raisin_Dataset.arff')
df = pd.DataFrame(data)
atributos = df.iloc[:,:-1]
classes = df.iloc[:,-1]
def calibrate_kmeans():
    print("calibrando kmeans: ")
    params = []
    scores=[]
    for c in range(2, 31):
        for i in range(10, 300, 20):
            km = KMeans(c, max_iter=i)
            params.append({"c":c, "n": i})
            km=km.fit(atributos)
            score=get_coe(km)
            score=get_sep(km, score)
            score=get_silhouette_score(atributos, km, score)
            score=get_scores(classes, km, score)
            scores.append(score)
    print(get_best(params, normalize_scores(scores)))

def calibrate_dbscan():
    print("calibrando dbscan: ")
    params = []
    scores=[]
    for eps in np.arange(10, 10000, 10):
        for ms in range(2, 10, 2):
            db = DBSCAN(eps=eps, min_samples=ms)
            db=db.fit(atributos)
            if(len(db.labels_[db.labels_!=-1])>=700) and len(set(db.labels_[db.labels_!=-1]))>1:
                params.append({"eps":eps, "ms": ms})
                score=get_coe(db, False, atributos)
                score=get_sep(db, score, False, atributos)
                score=get_silhouette_score(atributos, db, score)
                score=get_scores(classes, db, score)
                scores.append(score)
    print(get_best(params, normalize_scores(scores)))

def calibrate_agnes():
    print("calibrando agnes:")
    params = []
    scores=[]
    for c in range(2, 31, 1):
        for l in ["ward", "complete", "average", "single"]:
            agnes = AgglomerativeClustering(n_clusters=c, linkage=l)
            agnes=agnes.fit(atributos)
            params.append({"n_clusters":c, "l": l})
            score=get_coe(agnes, False, atributos)
            score=get_sep(agnes, score, False, atributos)
            score=get_silhouette_score(atributos, agnes, score)
            score=get_scores(classes, agnes, score)
            scores.append(score)
    print(get_best(params, normalize_scores(scores)))
#calibrate_kmeans()
#calibrate_dbscan()
calibrate_agnes()