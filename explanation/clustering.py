import argparse
import numpy as np
import pickle
from tqdm import tqdm
import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from batch_similarity import __semantic_callable


def load_embeddings(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_clustering(cl, fname):
    with open(fname, 'wb') as f:
        pickle.dump(cl, f)


def agglomerative_clustering(k, enembs, method, dest):
    """
    k: Set of values to try for k. For instance if you want to create 8, 10, 15 clusters, k=[8, 10, 15]
    enembs: Entity embeddings
    method: Algorithm used to compute the distance between embedding. More info here, at the 'affinity' field --> https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    """
    cl = AgglomerativeClustering(n_clusters=k, affinity=method, linkage='ward').fit(enembs)
    save_clustering(cl, "{}/agglomerative/{}/{}.pkl".format(dest, method, k))


def kmeans_clustering(k, enembs, dest):
    """
    k: How many clusters to create
    enembs: Entity embeddings
    dest: Folder where the clustering models will be saved
    """
    clustering = KMeans(n_clusters=k, algorithm="auto").fit(enembs)
    save_clustering(clustering, "{}/KMeans_{}.pkl".format(dest, k))


def create_distance_matrix_semantic(embs):
    """
    This function creates a distance matrix between embeddings based on the semantic score between them. It was intended
    to be used by the agglomerative clustering algorithm in order to perform clustering based  on the semantic distance
    between embeddings. At the moment it hasn't been implemented
    """
    with open("../datasets/DBpedia15k/entity2class_dict.pkl", 'rb') as f:
        classes = pickle.load(f)
    mat = np.zeros((len(embs), len(embs)))
    for i in tqdm(range(len(embs))):
        for j in range(len(embs)):
            if i != j and mat[i, j] == 0:
                dist = __semantic_callable(embs[i], embs[j], i, j, classes)
                mat[i, j] = dist
                mat[j, i] = dist
    return mat


def main(args):
    path_to_entity_embeddings = args.path_to_embs
    k = args.k
    dest = args.dest
    type = args.type
    method = args.method

    enembs = load_embeddings(path_to_entity_embeddings)
    if type == "kmeans":
        kmeans_clustering(k, enembs, dest)
    elif type == "agglomerative":
        agglomerative_clustering(k, enembs, method, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_embs", required=True, type=str, help="Path to the entity embeddings")
    parser.add_argument("--k", required=True, type=int, help="How many clusters to create")
    parser.add_argument("--dest", required=True, type=str, help="Path where the model will be saved")
    parser.add_argument("--type", required=True, type=str, choices=["kmeans", "agglomerative"])
    parser.add_argument("--method", required=False, type=str, help="This argument is used only when performing agglomerative clustering")
    args = parser.parse_args()
    now = time.time()
    main(args)
    print("Elapsed time: {}".format(time.time()-now))
