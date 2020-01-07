from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import pandas.plotting
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def task1():
    X = pd.read_csv("pluton.csv")

    for numIterations in [1, 2]:

        kmeans = KMeans(n_clusters=3, random_state=0, max_iter=numIterations).fit(X)

        colormap = plt.get_cmap('hsv')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
        axes = pd.plotting.scatter_matrix(X, color=colormap(norm(kmeans.labels_)))
        plt.suptitle(f'max_iter = {numIterations}')

        labels = kmeans.labels_
        print(f'max_iter = {numIterations}, n_iter = {kmeans.n_iter_}')
        print('Silhouette-Score = ', metrics.silhouette_score(X, labels, metric='euclidean'))
        print('Calinski-Harabaz Index = ', metrics.calinski_harabaz_score(X, labels))
        print('Davies-Bouldin Index', metrics.davies_bouldin_score(X, labels), end='\n\n')

    matplotlib.pyplot.show()


def task2():

    centers = [[5, 4], [10, 4.5], [2, 6]]
    X, y = make_blobs(n_samples=300, n_features=2, centers=centers, cluster_std=0.2, random_state=5)

    for i in range(X.shape[0]):
        distance_to_center = X[i][y[i]%2] - centers[y[i]][y[i]%2]
        X[i][y[i]%2] += 10 * distance_to_center

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("ground truth")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    from ml5_ import kmedoids
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score

    for metric in ['euclidean', 'manhattan']:

        D = pairwise_distances(X, metric=metric)
        clusters, medoids = kmedoids.cluster(D, 3)

        color_dict = {medoids[0]: 0, medoids[1]: 1, medoids[2]: 2}
        clusters = [color_dict[y] for y in clusters]
        print(f'k-means, metric = {metric}')
        print(f'homogeneity-score = {homogeneity_score(y, clusters)}')
        print(f'completeness-score = {completeness_score(y, clusters)}')
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusters, marker='o')
        plt.title(f'k-medoids, metric={metric}')


        D_scaled = pairwise_distances(X_scaled, metric=metric)
        clusters, medoids = kmedoids.cluster(D_scaled, 3)

        color_dict = {medoids[0]: 0, medoids[1]: 1, medoids[2]: 2}
        clusters = [color_dict[y] for y in clusters]
        print(f'k-means, metric = {metric}, scaled')
        print(f'homogeneity-score = {homogeneity_score(y, clusters)}')
        print(f'completeness-score = {completeness_score(y, clusters)}')
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusters, marker='o')
        plt.title(f'k-medioids, metric={metric}, scaled')

def task3():
    df = pd.read_csv('votes.csv')

    df = df.fillna(0)
    print(df)
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

    Z = linkage(df, method='ward')

    plt.figure()
    dendrogram(Z)
    plt.show()


#task1()
task2()
#task3()

plt.show()