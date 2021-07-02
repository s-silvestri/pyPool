#Todo: algoritmo di clusterizzazione: parti con tre spezzoni consecutivi da 0.3 overlappati di 0.1 (da 0 a 0.2, da 0.1 a 0.3, da 0.2 a 0.4), facci la yule-walker ed estrai i parametri rilevanti.  L'idea di fondo è che così si vede anche un minimo di evoluzione temporale del segnale. Ricrea i dataset con questi parametri, possibilmente accorpando più di una run e facendo di nuovo un centinaio di campioni estratti random. L'ipotesi di lavoro è la seguente: i tratti di strada belli sono molto più omogenei, quelli brutti ogni tanto hanno delle feature che sembrano belle ma non si confermano per 3 pezzi consecutivi. Inoltre, un outlier bello può rendere una media relativamente bella, mentre nell'evoluzione temporale la cosa la capisci. La clusterizzazione quindi dovrebbe funzionare meglio. Alla fine ti prendi la potenza e il rapporto di risonanze, quelli sono parametri buoni, in più la loro evoluzione temporale cioè il rapporto tra x_i e x_medio per un totale di 8 parametri

from sklearn.cluster import KMeans
#from keras.datasets import mnist
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

#Classificatori forti: Total power average, Ratio  1res average, Ratiohifreq_average (tra 1-2 e 0), Ci sono evoluzioni diverse di Ratio_1res e ratio_hifreq, tendenzialmente che crescono alla fine nelle due classi diverse, probabilmente il punto è che trattandosi di segnali comunque non "fissi" nelle classi 2 e 3 ci metta l'ingresso e l'uscita dalla strada brutta. Con 4 classi ti identifica tutto, ma l'algoritmo è molto instabile e da classificazioni diverse da una volta all'altra, probabilmente fa overfitting. Ci vogliono più dari.
loadir="F:/"

#data=pd.read_csv(loadir+'dataset_millenavacchio2.csv')
data=pd.read_csv('F:/tridataset.csv')
data=data.drop(columns=['Unnamed: 0'])
n_clusters=3
#col=data.loc[: , "Total_power1","Total_power2", "Total_power3"]
data.mean()
data['Total_power_avg'] = data[["Total_power1","Total_power2", "Total_power3"]].mean(axis=1)
data['Ratio1res_avg'] = data[["Ratio_1res1","Ratio_1res2", "Ratio_1res3"]].mean(axis=1)
data['Ratiohifreq_avg'] = data[["Ratio_hifreq1","Ratio_hifreq2", "Ratio_hifreq2"]].mean(axis=1)
data['Ratio_1res1']=data['Ratio_1res1']/data['Ratio1res_avg']
data['Ratio_1res2']=data['Ratio_1res2']/data['Ratio1res_avg']
data['Ratio_1res3']=data['Ratio_1res3']/data['Ratio1res_avg']
data['Total_power1']=data['Total_power1']/data['Total_power_avg']
data['Total_power2']=data['Total_power2']/data['Total_power_avg']
data['Total_power3']=data['Total_power3']/data['Total_power_avg']
data['Ratio_hifreq1']=data['Ratio_hifreq1']/data['Ratiohifreq_avg']
data['Ratio_hifreq2']=data['Ratio_hifreq2']/data['Ratiohifreq_avg']
data['Ratio_hifreq3']=data['Ratio_hifreq3']/data['Ratiohifreq_avg']
data=data.drop(columns=['Ratio_2res1','Ratio_2res2','Ratio_2res3','Ratio_3res1','Ratio_3res2','Ratio_3res3','Firstres1','Firstres2','Firstres3'])





X=data.drop(columns=['Label'])
y=data['Label'].values
X_train, X_test, y_train, y_test=train_test_split(X,y, stratify=y, test_size=0.7, random_state=1987)
good_init=np.array([[50,80,6],[400,25,2.5]])
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(X)
dbscan = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(X)

gm = GaussianMixture(n_components=2, n_init=10)
gm.fit(X)
bgm = BayesianGaussianMixture(n_components=2, n_init=10)
bgm.fit(X)
print (X.columns)
print (kmeans.cluster_centers_)

# #nca = NeighborhoodComponentsAnalysis(random_state=87)
# #knn = KNeighborsClassifier(n_neighbors=3)
# #nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
# #nca_pipe.fit(X_train, y_train)
# #print(nca_pipe.score(X_test, y_test))
# #return (nca_pipe)
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))
# x = x.reshape((x.shape[0], -1))
# x = np.divide(x, 255.)
# # 10 clusters
# n_clusters = len(np.unique(y))
# # Runs in parallel 4 CPUs
# kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
# # Train K-Means.
# y_pred_kmeans = kmeans.fit_predict(x)
# # Evaluate the K-Means clustering accuracy.
# metrics.acc(y, y_pred_kmeans)


