# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi

# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

"""
Obtén el coeficiente s de X para diferente número de vecindades 
k ∈ {1, 2, 3, ..., 15} utilizando el algoritmo KMeans. Muestra gráficamente s 
en función de k y decide cuál es el número óptimo de vecindades.
"""

# ################################# K-MEANS ###################################
print (' K-MEANS '.center(80,'#'))
# Representamos los valores del coef. de Silhouette frente al número de 
# clusters k. Para un solo cluster tomamos el valor s=-1 (evaluando en la fórmula)

n_clusters_values, silhouette_values = np.arange(1,16), [-1]

for num_clusters in range (2,16): # Creamos el vector de coefs. de Silhouette
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    silhouette = metrics.silhouette_score(X, kmeans.labels_)
    silhouette_values.append(silhouette)
silhouette_values = np.array(silhouette_values)

fig, ax = plt.subplots()
ax.plot(n_clusters_values, silhouette_values, 'ro--')
ax.set(xlabel='Number of clusters', ylabel='Silhouette')
ax.grid()
plt.show()

print ('The maximum Silhouette Coef. is obtained with %d clusters'
       % n_clusters_values[silhouette_values.argmax()])

# Análisis de Silhouette y gráfico de dispersión para cada valor de n_clusters_
# con el fin de determinar de forma más clara el número de clusters k óptimo

plt.style.use(plt.style.available[1])
n_clusters_posibles = range (2, 6)

for n_clusters_ in n_clusters_posibles:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

    kmeans_ = KMeans(n_clusters=n_clusters_, random_state=0).fit(X)
    labels_ = kmeans_.labels_
    
    silhouette_ = metrics.silhouette_score(X, labels_) 
    elements_silhouette = metrics.silhouette_samples(X, labels_) # Coef. de Silh.
                                                                 # para cada muestra
    y_lower = 10 # Coordenada y de inicio de la franja correspondiente al cluster
    unique_labels_ = set(labels_)
    colors_ = [plt.cm.Spectral(each) 
               for each in np.linspace(0, 1, len(unique_labels_))]

    for k, col in zip(unique_labels_, colors_):
        if k == -1:
            col = [0, 0, 0, 1] # Negro para los outliers

        xy = X[labels_ == k]
        # Tomamos los coefs. de Silhouette de cada elemento del cluster y ordenamos
        xy_elements_silhouette = elements_silhouette[labels_==k]
        xy_elements_silhouette.sort() 

        y_upper = y_lower + len(xy) # Coordenada y del final de la franja
        
        ax1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), 
                 markeredgecolor='k', markersize=5)
    
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, 
                          xy_elements_silhouette, facecolor=tuple(col), 
                          edgecolor=tuple(col), alpha=1)   
        ax2.text(-0.05, y_lower + 0.5 * len(xy), str(k))
        
        y_lower = y_upper + 10 
    
    ax2.axvline(silhouette, color="red", linestyle="--", lw=1.5)
    ax2.legend(['s=%0.3f' % silhouette_])
    ax2.set(xlabel='Shilhouette Coefficients', ylabel='Elements by clusters',
            yticklabels=([]), 
            xlim=[-0.1,1], ylim=[0, len(X) + (n_clusters_ + 1)*10])
    
    ax1.set_xlim(-2.5, 2.2), ax1.set_ylim(-2.3, 2.3)
    
    plt.suptitle('Clusters: %s\nSilhouette: %0.3f'
                 % (n_clusters_, silhouette_), fontsize=11, fontweight='bold')
    plt.show()
    
    print('CLUSTERS CENTERS:')
    print(kmeans_.cluster_centers_)
    print("SILHOUETTE COEFFICIENT: %0.3f".upper() % silhouette_)

plt.style.use('default')

# Resultado con k = 3 (óptimo)

kmeans_ = KMeans(n_clusters=3, random_state=0)
kmeans_.fit(X)
vor = Voronoi(kmeans_.cluster_centers_)

fig = plt.figure(figsize=(8,4))
plt.scatter(X[:, 0], X[:, 1], marker='o',
            c=kmeans_.predict(X), s=5, cmap='summer')
plt.title('Optimal choice: k = 3')
plt.show()

"""
Obtén el coefi
ciente s para el mismo sistema X usando ahora el algoritmo DBSCAN 
con la métrica euclidean y luego con manhattan. En este caso, el parámetro que 
debemos explorar es el umbral de distancia, de
jando el número de elementos 
mínimo en n0 = 10. Comparad gra
camente con el resultado del apartado anterior.
"""

############################# DBSCAN EUCLIDEAN ################################
print (' DBSCAN EUCLIDEAN METRIC '.center(80,'#'))
# Para eps > 0.43 surgen problemas con el cálculo del coef. de Silhouette,
# ya que las bolas de los elementos son lo suficientemente grandes como para 
# que resulte un sólo cluster.

epsilon_values = np.arange(.10, .44, .01) # tomamos valores de eps < 0.44
silhouette_values = []

# Representamos s en función de épsilon
for eps in epsilon_values:
    db = DBSCAN(eps=eps, min_samples=10, metric='euclidean').fit(X)
    silhouette_ = metrics.silhouette_score(X, db.labels_)
    silhouette_values.append(silhouette_)
silhouette_values_ = np.array(silhouette_values)
    
fig = plt.figure(figsize=(8,4))
plt.plot(epsilon_values, silhouette_values_, 'ro--', lw=1, markersize=5,
         markeredgecolor='k')
plt.title('Euclidean Metric', fontsize=12, fontweight='bold')
plt.xlabel('Epsilon'), plt.ylabel('Shilhouette Coefficient')
plt.grid()
plt.show()

epsilon = epsilon_values[silhouette_values_.argmax()] # Tomamos el épsilon con
                                                      # mayor coef. silhouette

print ('Umbral de distancia elegido: epsilon = %0.2f' % epsilon)
print()

# Aplicamos el algoritmo con el épsilon óptimo elegido

db = DBSCAN(eps=0.24, min_samples=10, metric='euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % silhouette_values_.max()) 

# Representamos el resultado con un plot

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
plt.show()


# ########################### DBSCAN MANHATTAN ################################
print (' DBSCAN MANHATTAN METRIC '.center(80,'#'))
# Para eps > 0.51 surgen problemas, ya que nos conduce a clasificar los puntos 
# en un sólo cluster y no puede calcular el coef. de Silhouette

epsilon_values = np.arange(.10, .52, .01) # epsilon < 0.53
silhouette_values = []

# Representamos los coeficientes de Silhouette en función del umbral distancia

for eps in epsilon_values:
    db = DBSCAN(eps=eps, min_samples=10, metric='manhattan').fit(X)
    silhouette_ = metrics.silhouette_score(X, db.labels_)
    silhouette_values.append(silhouette_)
silhouette_values_ = np.array(silhouette_values)
    
fig = plt.figure(figsize=(10,4))
plt.plot(epsilon_values, silhouette_values_, 'ro--', lw=1, markersize=5,
         markeredgecolor='k')
plt.title('Manhattan Metric', fontsize=12, fontweight='bold')
plt.xlabel('Épsilon'), plt.ylabel('Shilhouette Coefficient')
plt.grid()
plt.show()

epsilon = epsilon_values[silhouette_values_.argmax()] # Tomamos el épsilon cuyo
                                                      # silhouette es más elevado  

print ('Chosen distance threshold: epsilon = %0.2f' % epsilon)
print()

# Aplicamos el algoritmo, esta vez con el épsilon óptimo

db = DBSCAN(eps=epsilon, min_samples=10, metric='manhattan').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % silhouette_values_.max())

# Representamos el resultado con un plot

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
plt.show()


"""
Se intentó aplicar un método para la elección de épsilon basado en la curvatura
de la gráfica que pinta el código de abajo sin llegar a resultados concluyentes.

neigh = NearestNeighbors(n_neighbors=2, metric='manhattan')
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
fig = plt.figure()
plt.plot(distances)
plt.grid()
plt.show()
"""
