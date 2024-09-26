import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.cluster import k_means
from sklearn.cluster import dbscan
from sklearn.cluster import AgglomerativeClustering
data, meta = arff.loadarff('Raisin_Dataset.arff')
df = pd.DataFrame(data)
print("dataset original\n")
df.info()
atributos = df.iloc[:,:-1]
classes = df.iloc[:,-1]
