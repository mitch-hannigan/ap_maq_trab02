import pandas as pd
import numpy as np
from scipy.io import arff
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
max_iters=20
def soma(cf1, cf2, cf3, cf4, cf5, atributos):
    proba_cf1 = cf1.predict_proba(atributos)
    proba_cf2 = cf2.predict_proba(atributos)
    proba_cf3 = cf3.predict_proba(atributos)
    proba_cf4 = cf4.predict_proba(atributos)
    proba_cf5 = cf5.predict_proba(atributos)
    sum_proba = (proba_cf1 + proba_cf2 + proba_cf3+proba_cf4+proba_cf5)
    return np.argmax(sum_proba, axis=1)
def voto(cf1, cf2, cf3, cf4, cf5, atributos):
    prediction_cf1 = cf1.predict(atributos)
    prediction_cf2 = cf2.predict(atributos)
    prediction_cf3 = cf3.predict(atributos)
    prediction_cf4 = cf4.predict(atributos)
    prediction_cf5 = cf5.predict(atributos)
    return np.array([np.bincount([prediction_cf1[i], prediction_cf2[i], prediction_cf3[i], prediction_cf4[i], prediction_cf5[i]]).argmax() for i in range(len(atributos))])
def borda_count(cf1, cf2, cf3, cf4, cf5, atributos):
    proba_cf1 = cf1.predict_proba(atributos)
    proba_cf2 = cf2.predict_proba(atributos)
    proba_cf3 = cf3.predict_proba(atributos)
    proba_cf4 = cf4.predict_proba(atributos)
    proba_cf5 = cf5.predict_proba(atributos)
    ranks = np.array([rankdata(proba, axis=1) for proba in [proba_cf1, proba_cf2, proba_cf3, proba_cf4, proba_cf5]])
    borda_scores = np.sum(ranks, axis=0)
    return np.argmax(borda_scores, axis=1)



data, meta = arff.loadarff('Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data)
print("dataset original\n")
df.info()
acc_knn, acc_mlp, acc_dt, acc_svm, acc_nb, acc_soma, acc_voto, acc_borda_count=list(),list(),list(),list(),list(), list(),list(),list()
for i in range(0, max_iters):
    print("iteração ", i+1)
    df = shuffle(df)
    atributos = df.iloc[:,:-1]
    classes = df.iloc[:,-1]
    atributos_treino,atributos_temp,classes_treino,classes_temp=train_test_split(atributos,classes,test_size=0.5,stratify=classes)
    atributos_validacao,atributos_ttestes,classes_validacao,classes_testes=train_test_split(atributos_temp,classes_temp,test_size=0.5,stratify=classes_temp)
    best_naive_bayes = GaussianNB()
    best_naive_bayes.fit(atributos_treino, classes_treino)
    opiniao = best_naive_bayes.predict(atributos_validacao)
    temp_acc = accuracy_score(classes_validacao, opiniao)
    print("validação, a acurácia do nayve_bays foi de ", temp_acc*100, "%")
    acc=0.0
    best_knn=None
    best_peso, best_k = None, None
    for peso in ("distance", "uniform"):
        for k in range(1, 51):
            temp = KNeighborsClassifier(n_neighbors=k, weights=peso)
            temp.fit(atributos_treino, classes_treino)
            opiniao = temp.predict(atributos_validacao)
            temp_acc = accuracy_score(classes_validacao, opiniao)
            if(temp_acc>acc):
                best_k = k
                best_peso = peso
                best_knn = temp
                acc=temp_acc
    print("validação, os melhores parâmetros para o knn nessa iteração são:\nK = ", best_k, "\npeso = ", best_peso, "\n", "acurácia = ", acc*100, "%\n")
    acc=0.0
    best_svm=None
    best_k, best_c = None, None
    for k in ("linear", "poly", "rbf", "sigmoid"):
        for c in range(1, 5):
            temp = SVC(kernel=k, C=c, probability=True)
            temp.fit(atributos_treino, classes_treino)
            opiniao = temp.predict(atributos_validacao)
            temp_acc = accuracy_score(classes_validacao, opiniao)
            if(temp_acc>acc):
                best_k = k
                best_c = c
                best_svm = temp
                acc=temp_acc
    print("validação, os melhores parâmetros para o SVM nessa iteração são:\nKernel = ", best_k, "\nC = ", best_c, "\n", "acurácia = ", acc*100, "%\n")
    acc=0.0
    best_d_t =None
    best_crit, best_d, best_mss, best_msl = 0, 0, 0, 0
    for crit in ("gini", "entropy"):
        for d in range(3, 20):
            for msl in range(1, 6):
                for mss in range(msl*2, msl*4+1):
                    temp = DecisionTreeClassifier(criterion=crit, max_depth=d, min_samples_leaf=msl, min_samples_split=mss)
                    temp.fit(atributos_treino, classes_treino)
                    opiniao = temp.predict(atributos_validacao)
                    temp_acc = accuracy_score(classes_validacao, opiniao)
                    if(temp_acc>acc):
                        best_crit = crit
                        best_d = d
                        best_mss = mss
                        best_msl = msl
                        best_d_t = temp
                        acc=temp_acc
    print("validação, os melhores parâmetros para a árvore de decisão nessa iteração são:\ncriterion = ", best_crit, "\nmax_depth = ", best_d, "\nmin_samples_leaf =", best_msl, "\nmin_samples_split =", best_mss, "\n", "acurácia = ", acc*100, "%\n")
    acc=0.0
    best_mlp =None
    best_neurons, best_t_lr, best_its, best_act = 0, 0, 0, 0
    for neurons in (6, 12, 25, 50, 100):
        for t_lr in ("constant", "invscaling", "adaptive"):
            for its in (150, 300, 500, 1000):
                for act in ("identity", "logistic", "tanh", "relu"):
                    temp = MLPClassifier(activation=act, learning_rate=t_lr, max_iter=its, hidden_layer_sizes=(neurons, neurons, neurons))
                    temp.fit(atributos_treino, classes_treino)
                    opiniao = temp.predict(atributos_validacao)
                    temp_acc = accuracy_score(classes_validacao, opiniao)
                    if(temp_acc>acc):
                        best_mlp = temp
                        best_act = act
                        best_t_lr=t_lr
                        best_neurons = neurons
                        best_its = its
                        acc=temp_acc
    print("validação, os melhores parâmetros para o MLP nesta iteração são:\niterações = ", best_its, "\nneurons (3 camadas) = ", best_neurons, "\nativação =", best_act, "\ntaxa de aprendizagem =", best_t_lr, "\n", "acurácia = ", acc*100, "%\n")
    print("resultado das execuções no conjunto de testes")
    opiniao = best_naive_bayes.predict(atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_nb.append(acc)
    print("O naive bayes obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = best_knn.predict(atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_knn.append(acc)
    print("O KNN obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = best_svm.predict(atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_svm.append(acc)
    print("O SVM obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = best_d_t.predict(atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_dt.append(acc)
    print("A árvore de decisão obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = best_mlp.predict(atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_mlp.append(acc)
    print("O MLP obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = soma(best_naive_bayes, best_knn, best_svm, best_d_t, best_mlp, atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_soma.append(acc)
    print("A regra da soma obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = voto(best_naive_bayes, best_knn, best_svm, best_d_t, best_mlp, atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_voto.append(acc)
    print("O voto majoritário obteve", acc*100, "% de acurácia nesta iteração.")
    opiniao = borda_count(best_naive_bayes, best_knn, best_svm, best_d_t, best_mlp, atributos_ttestes)
    acc = accuracy_score(classes_testes, opiniao)
    acc_borda_count.append(acc)
    print("O método de borda count obteve", acc*100, "% de acurácia nesta iteração.")
#fazer análize estatística