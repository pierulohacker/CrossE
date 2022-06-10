import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
import umap


def load_embeddings(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_clustering(cl, fname):
    with open(fname, 'wb') as f:
        pickle.dump(cl, f)


def agglomerative_clustering(k, enembs, method='euclidean', linkage='ward'):
    for i in k:
        cl = AgglomerativeClustering(n_clusters=i, affinity=method, linkage=linkage).fit(enembs)
        save_clustering(cl, "../Agglomerative_{}.pkl".format(i))


def kmeans_clustering(k, enembs, method="auto"):
    for i in k:
        clustering = KMeans(n_clusters=i, algorithm=method).fit(enembs)
        save_clustering(clustering, "KMeans_{}_{}.pkl".format(i, method))


if __name__ == "__main__":
    enembs = load_embeddings('../save/DBpedia15k/out_data/pickle/DBPedia_selected.pkl')
    r = [8,10,15]
    #kmeans_clustering(r, enembs)
    agglomerative_clustering(r, enembs, 'manhattan', 'average')
    #agglomerative_clustering(r, enembs)
    #agglomerative_clustering(r, enembs, "cosine", "average")
#python3 explanation/explanation.py --data ./save/DBpedia15k/out_data/pickle/ --save_dir explanation/results/DBpedia15k/euclidian/agglomerative8/2perc/ --multiprocessing True --clustering agglomerative-8
