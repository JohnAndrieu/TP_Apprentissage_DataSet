import numpy
from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import time
from sklearn.preprocessing import StandardScaler
import re

file_smile = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/smile1.arff"
x1 = "/home/jandrieu/Bureau/dataset/x1.txt"
x2 = "/home/jandrieu/Bureau/dataset/x2.txt"
x3 = "/home/jandrieu/Bureau/dataset/x3.txt"
x4 = "/home/jandrieu/Bureau/dataset/x4.txt"
y1 = "/home/jandrieu/Bureau/dataset/y1.txt"


def save_fig(x, y, labels, name):
    plt.figure()
    plt.scatter(x, y, c=labels, marker='.', s=0.0005)
    plt.savefig(name)


def save_fig_nolabels(x, y, name):
    plt.figure()
    plt.scatter(x, y, marker='.', s=0.0005)
    plt.savefig(name)


def erase_file(file_path):
    file = open(file_path, "w")
    file.close()


def insert_section(file_path, section_name):
    f = open(file_path, "a")
    f.write("\n###### ###### " + section_name + " ###### ######\n\n")
    f.close()


def extract_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(re.sub(' +', ' ', line.strip()).split(' '))
    return np.array(data, float)


def visualisation():
    x1_train = extract_data(x1)
    x2_train = extract_data(x2)
    x3_train = extract_data(x3)
    x4_train = extract_data(x4)
    y1_train = extract_data(y1)
    save_fig_nolabels(x1_train[:, 0], x1_train[:, 1], './raw_visualisation/x1')
    save_fig_nolabels(x2_train[:, 0], x2_train[:, 1], './raw_visualisation/x2')
    save_fig_nolabels(x3_train[:, 0], x3_train[:, 1], './raw_visualisation/x3')
    save_fig_nolabels(x4_train[:, 0], x4_train[:, 1], './raw_visualisation/x4')
    save_fig_nolabels(y1_train[:, 0], y1_train[:, 1], './raw_visualisation/y1')


def run_KMeans(nb_cluster, data, label):
    tmps1 = time.time()
    kmeans = KMeans(n_clusters=nb_cluster, init='k-means++')
    kmeans.fit(data)
    tmps2 = time.time() - tmps1
    coeff_silhou = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')  # doit être grand
    coeff_davies = davies_bouldin_score(data, kmeans.labels_)  # doit être petit
    f = open("./execution_time/kmeans_clustering/kmeans_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    msg_silhou = "Coefficient de Silhouette = %f\n" % coeff_silhou
    msg_davies = "Coefficient de Davies = %f\n" % coeff_davies
    f.write(msg_time + msg_silhou + msg_davies)
    f.close()
    plt.scatter(nb_cluster, coeff_silhou, c='red', marker='.')
    plt.scatter(nb_cluster, coeff_davies, c='blue', marker='x')
    return kmeans


def runAndSave_KMeans(nb_cluster, data, x, y, name, name_fig):
    kmeans = run_KMeans(nb_cluster, data, "Kmeans- " + name + " [" + str(nb_cluster) + "]")
    save_fig(x, y, kmeans.labels_, "./kmeans_graph/" + name_fig)


def iter_KMeansClustering(data, name):
    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans- " + name)
    plt.figure()
    for iter_cluster in range(5, 20):
        run_KMeans(iter_cluster, data, "Kmeans- " + name + " [" + str(iter_cluster) + "]")
    plt.savefig("./metrics/kmeans/" + name)


def runClustering_KMeans(data, name):
    data_train = extract_data(data)
    nb_cluster = 35
    save_fig_nolabels(data_train[:, 0], data_train[:, 1], "./kmeans_graph/" + name)
    runAndSave_KMeans(nb_cluster, data_train, data_train[:, 0], data_train[:, 1], name, name + "_kmeans")
    iter_KMeansClustering(data_train, name)


def Clustering_KMeans():
    erase_file("./execution_time/kmeans_clustering/kmeans_clustering.txt")
    """
    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans x1"
                   + " nombre clusters fixé")

    print("x1")
    runClustering_KMeans(x1, "x1")

    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans x2"
                   + " nombre clusters fixé")
    print("x2")
    runClustering_KMeans(x2, "x2")

    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans x3"
                   + " nombre clusters fixé")
    print("x3")
    runClustering_KMeans(x3, "x3")

    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans x4"
                   + " nombre clusters fixé")
    print("x4")
    runClustering_KMeans(x4, "x4")
    """
    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans y1"
                   + " nombre clusters fixé")

    print("y1")
    runClustering_KMeans(y1, "y1")


def run_AggloClustering(nb_cluster, data_train, linkage, label):
    tmps1 = time.time()
    agglo = AgglomerativeClustering(nb_cluster, linkage=linkage)
    agglo.fit(data_train)
    tmps2 = time.time() - tmps1
    coeff_silhou = metrics.silhouette_score(data_train, agglo.labels_, metric='euclidean')  # doit être grand
    coeff_davies = davies_bouldin_score(data_train, agglo.labels_)  # doit être petit
    f = open("./execution_time/agglo_clustering/agglo_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    msg_silhou = "Coefficient de Silhouette = %f\n" % coeff_silhou
    msg_davies = "Coefficient de Davies = %f\n" % coeff_davies
    f.write(msg_time + msg_silhou + msg_davies)
    f.close()
    plt.scatter(nb_cluster, coeff_silhou, c='red', marker='.')
    plt.scatter(nb_cluster, coeff_davies, c='blue', marker='x')
    return agglo


def iter_AggloClustering(data, linkage, name):
    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering [" + name +
                   "] [" + linkage + "]")
    plt.figure()
    for iter_cluster in range(2, 10):
        run_AggloClustering(iter_cluster, data, linkage,
                            "Agglomeratif " + linkage + "- " + name + " [" + str(iter_cluster) + "] [" + linkage + "]")
    plt.savefig("./metrics/agglomeratif/" + name)


def runAndSave_Agglo(nb_cluster, data, linkage, name, x, y, name_fig):
    agglo = run_AggloClustering(nb_cluster, data, linkage,
                                "Agglomeratif " + linkage + "- " + name + " [" + str(
                                    nb_cluster) + "] [" + linkage + "]")
    save_fig(x, y, agglo.labels_, "./agglomeratif_graph/" + name_fig)


def runClustering_Agglomeratif(nb_cluster, file, name):
    data = extract_data(file)
    runAndSave_Agglo(nb_cluster, data, "single", name, data[:, 0], data[:, 1], name + "_single")
    runAndSave_Agglo(nb_cluster, data, "average", name, data[:, 0], data[:, 1], name + "_average")
    runAndSave_Agglo(nb_cluster, data, "complete", name, data[:, 0], data[:, 1], name + "_complete")
    runAndSave_Agglo(nb_cluster, data, "ward", name, data[:, 0], data[:, 1], name + "_ward")
    iter_AggloClustering(data, "single", name)
    iter_AggloClustering(data, "average", name)
    iter_AggloClustering(data, "complete", name)
    iter_AggloClustering(data, "ward", name)


def Clustering_Agglomeratif():
    """
    erase_file("./execution_time/agglo_clustering/agglo_clustering.txt")
    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering x1"
                   + " nombre clusters fixé")

    nb_cluster = 15
    runClustering_Agglomeratif(nb_cluster, x1, "x1")

    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering x2"
                   + " nombre clusters fixé")

    runClustering_Agglomeratif(nb_cluster, x2, "x2")

    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering x3"
                   + " nombre clusters fixé")

    runClustering_Agglomeratif(nb_cluster, x3, "x3")

    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering x4"
                   + " nombre clusters fixé")

    runClustering_Agglomeratif(nb_cluster, x4, "x4")

    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering y1"
                   + " nombre clusters fixé")
    """
    nb_cluster = 15
    runClustering_Agglomeratif(nb_cluster, y1, "y1")


def run_DBSCANClustering(distance, min_pts, data_train, label):
    tmps1 = time.time()
    data_train = StandardScaler().fit_transform(data_train)
    dbscan = DBSCAN(eps=distance, min_samples=min_pts).fit(data_train)
    tmps2 = time.time() - tmps1
    labels = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    msg_clusters = 'Estimated number of clusters: %d' % n_clusters_
    msg_noise = 'Estimated number of noise points: %d' % n_noise_
    f = open("./execution_time/dbscan_clustering/dbscan_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    f.write(msg_time + msg_clusters + "\n" + msg_noise + "\n")
    f.close()
    return dbscan


def runAndSave_DBSCAN(distance, min_pts, data, name, x, y, name_fig):
    dbscan = run_DBSCANClustering(distance, min_pts, data, "DBSCAN - " + name + " - dt[" + str(distance)
                                  + "] pts[" + str(min_pts) + "]")
    save_fig(x, y, dbscan.labels_, "./dbscan_graph/" + name_fig)
    print("./dbscan_graph/" + name_fig)


def iter_DBSCANClustering(data, name, x, y):
    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN Clustering [" + name + "]")
    for distance in numpy.linspace(0.1, 0.2, 20):
        print(distance)
        for samples in range(2, 10):
            distance = round(distance, 1)
            runAndSave_DBSCAN(distance, samples, data, name, x, y, name + "_" + "dt[" + str(distance).replace('.', ',')
                              + "]_pts[" + str(samples) + "]")


def runClustering_DBSCAN(filename, distance, min_pts, name):
    data_train = extract_data(filename)
    runAndSave_DBSCAN(distance, min_pts, data_train, name, data_train[:, 0], data_train[:, 1], name)


def Clustering_DBSCAN():
    erase_file("./execution_time/dbscan_clustering/dbscan_clustering.txt")
    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN x1"
                   + " distance et nombre de points fixés")
    distance = 5
    min_pts = 0.5
    runClustering_DBSCAN(x1, distance, min_pts, "x1")
    data = extract_data(x1)
    iter_DBSCANClustering(data, "x1", data[:, 0], data[:, 1])

    """
    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN x2"
                   + " distance et nombre de points fixés")

    distance = 0.35
    min_pts = 14
    runClustering_DBSCAN(x2, distance, min_pts, "x2")
    data = extract_data(x2)
    iter_DBSCANClustering(data, "x1", data[:, 0], data[:, 1])

    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN x3"
                   + " distance et nombre de points fixés")

    distance = 5
    min_pts = 0.5
    runClustering_DBSCAN(x3, distance, min_pts, "x3")
    data = extract_data(x3)
    iter_DBSCANClustering(data, "x3", data[:, 0], data[:, 1])

    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN x4"
                   + " distance et nombre de points fixés")
    
    distance = 5
    min_pts = 0.5
    runClustering_DBSCAN(x4, distance, min_pts, "x4")
    data = extract_data(x4)
    iter_DBSCANClustering(data, "x4", data[:, 0], data[:, 1])

    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN y1"
                   + " distance et nombre de points fixés")

    distance = 5
    min_pts = 0.5
    runClustering_DBSCAN(y1, distance, min_pts, "y1")
    data = extract_data(y1)
    iter_DBSCANClustering(data, "y1", data[:, 0], data[:, 1])
    """

def run_HDBSCANClustering(data_train, label):
    tmps1 = time.time()
    data_train = StandardScaler().fit_transform(data_train)
    hdb = hdbscan.HDBSCAN(min_cluster_size=7).fit(data_train)
    tmps2 = time.time() - tmps1
    labels = hdb.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    msg_clusters = 'Estimated number of clusters: %d' % n_clusters_
    msg_noise = 'Estimated number of noise points: %d' % n_noise_
    f = open("./execution_time/hdbscan_clustering/hdbscan_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    f.write(msg_time + msg_clusters + "\n" + msg_noise + "\n")
    f.close()
    return hdb


def runAndSave_HDBSCAN(data, name, x, y, name_fig):
    dbscan = run_HDBSCANClustering(data, "HDBSCAN - " + name)
    save_fig(x, y, dbscan.labels_, "./hdbscan_graph/" + name_fig)
    print("./hdbscan_graph/" + name_fig)


def Clustering_HDBSCAN():
    erase_file("./execution_time/hdbscan_clustering/hdbscan_clustering.txt")
    insert_section("./execution_time/hdbscan_clustering/hdbscan_clustering.txt", "HDBSCAN x1"
                   + " distance et nombre de points fixés")

    data_train = extract_data(x1)
    runAndSave_HDBSCAN(data_train, "x1", data_train[:, 0], data_train[:, 1], "x1_hdbscan")

    insert_section("./execution_time/hdbscan_clustering/hdbscan_clustering.txt", "HDBSCAN x2"
                   + " distance et nombre de points fixés")

    data_train = extract_data(x2)
    runAndSave_HDBSCAN(data_train, "x2", data_train[:, 0], data_train[:, 1], "x2_hdbscan")

    insert_section("./execution_time/hdbscan_clustering/hdbscan_clustering.txt", "HDBSCAN x3"
                   + " distance et nombre de points fixés")

    data_train = extract_data(x3)
    runAndSave_HDBSCAN(data_train, "x3", data_train[:, 0], data_train[:, 1], "x3_hdbscan")

    insert_section("./execution_time/hdbscan_clustering/hdbscan_clustering.txt", "HDBSCAN x4"
                   + " distance et nombre de points fixés")

    data_train = extract_data(x4)
    runAndSave_HDBSCAN(data_train, "x4", data_train[:, 0], data_train[:, 1], "x4_hdbscan")

    insert_section("./execution_time/hdbscan_clustering/hdbscan_clustering.txt", "HDBSCAN y1"
                   + " distance et nombre de points fixés")

    data_train = extract_data(y1)
    runAndSave_HDBSCAN(data_train, "y1", data_train[:, 0], data_train[:, 1], "y1_hdbscan")


def main(param):
    if(param == "1"):
        visualisation()
    if (param == "2"):
        Clustering_KMeans()
    if (param == "3"):
        Clustering_Agglomeratif()
    if (param == "4"):
        Clustering_DBSCAN()
    if (param == "5"):
        Clustering_HDBSCAN()

"""
if __name__ == "__main__":
    main()
"""