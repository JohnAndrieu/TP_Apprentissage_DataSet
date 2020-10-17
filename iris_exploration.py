import time
from scipy.io import arff
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import hdbscan

iris = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/real-world/iris.arff"


def preparation():
    data_iris = arff.loadarff(open(iris, 'r'))
    df = pd.DataFrame(data_iris[0])
    setosa_df = df[df['class'] == b'Iris-setosa']
    virginica_df = df[df['class'] == b'Iris-virginica']
    versicolor_df = df[df['class'] == b'Iris-versicolor']
    return df, setosa_df, virginica_df, versicolor_df


def visualisation(setosa, virginica, versicolor):
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(21, 10))

    setosa.plot(x="sepallength", y="sepalwidth", kind="scatter", ax=ax[0], label='setosa', color='r')
    versicolor.plot(x="sepallength", y="sepalwidth", kind="scatter", ax=ax[0], label='versicolor', color='b')
    virginica.plot(x="sepallength", y="sepalwidth", kind="scatter", ax=ax[0], label='virginica', color='g')

    setosa.plot(x="petallength", y="petalwidth", kind="scatter", ax=ax[1], label='setosa', color='r')
    versicolor.plot(x="petallength", y="petalwidth", kind="scatter", ax=ax[1], label='versicolor', color='b')
    virginica.plot(x="petallength", y="petalwidth", kind="scatter", ax=ax[1], label='virginica', color='g')

    ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
    ax[1].set(title='Petal Comparasion', ylabel='petal-width')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('./iris/iris')


def kmeans(df):
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(21, 10))

    df_sepal = df[['sepallength', 'sepalwidth']]
    df_petal = df[['petallength', 'petalwidth']]

    tmps1 = time.time()
    kmeans_sepal = KMeans(n_clusters=3).fit(df_sepal)
    kmeans_petal = KMeans(n_clusters=3).fit(df_petal)
    tmps2 = time.time() - tmps1

    f = open("./iris/execution_time/kmeans_clustering.txt", "a")
    msg_time = "Temps d'execution = %f\n" % tmps2
    f.write(msg_time)
    f.close()

    ax[0].scatter(df['sepallength'], df['sepalwidth'], c=kmeans_sepal.labels_, marker='.')
    ax[1].scatter(df['petallength'], df['petalwidth'], c=kmeans_petal.labels_, marker='.')
    ax[0].set(title='Sepal comparasion ', ylabel='sepal-width', xlabel='sepal-length')
    ax[1].set(title='Petal Comparasion', ylabel='petal-width', xlabel='petal-length')
    plt.savefig('./iris/kmeans/iris_kmeans')


def run_agglomeratif(df, linkage):
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(21, 10))

    df_sepal = df[['sepallength', 'sepalwidth']]
    df_petal = df[['petallength', 'petalwidth']]

    tmps1 = time.time()
    agglo_sepal = AgglomerativeClustering(3, linkage=linkage).fit(df_sepal)
    tmps2 = time.time() - tmps1

    _tmps1 = time.time()
    agglo_petal = AgglomerativeClustering(3, linkage=linkage).fit(df_petal)
    _tmps2 = time.time() - _tmps1

    f = open("./iris/execution_time/agglo_clustering.txt", "a")
    msg_time = "Temps d'execution [sepal] [" + linkage + "] = %f\n" % tmps2
    _msg_time = "Temps d'execution [petal] [" + linkage + "] = %f\n" % _tmps2
    f.write(msg_time + _msg_time)
    f.close()

    ax[0].scatter(df['sepallength'], df['sepalwidth'], c=agglo_sepal.labels_, marker='.')
    ax[1].scatter(df['petallength'], df['petalwidth'], c=agglo_petal.labels_, marker='.')
    ax[0].set(title='Sepal comparasion ', ylabel='sepal-width', xlabel='sepal-length')
    ax[1].set(title='Petal Comparasion', ylabel='petal-width', xlabel='petal-length')
    plt.savefig('./iris/agglo/iris_agglo_' + linkage)


def agglomeratif(df):
    run_agglomeratif(df, 'single')
    run_agglomeratif(df, 'average')
    run_agglomeratif(df, 'complete')
    run_agglomeratif(df, 'ward')


def main():
    df, setosa, virginica, versicolor = preparation()
    #visualisation(setosa, virginica, versicolor)
    #kmeans(df)
    agglomeratif(df)


if __name__ == "__main__":
    main()
