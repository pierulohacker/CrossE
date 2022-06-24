import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from batch_similarity import __semantic_callable
import numpy as np
from tqdm import tqdm

def load_embeddings(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_clustering(cl, fname):
    with open(fname, 'wb') as f:
        pickle.dump(cl, f)


def agglomerative_clustering(k, enembs, method='euclidean', linkage='ward'):
    for i in k:
        cl = AgglomerativeClustering(n_clusters=i, affinity=method, linkage=linkage).fit(enembs)
        save_clustering(cl, "../agglomerative_DBP/{}/{}.pkl".format(method, i))


def kmeans_clustering(k, enembs, method="auto"):
    for i in k:
        clustering = KMeans(n_clusters=i, algorithm=method).fit(enembs)
        save_clustering(clustering, "KMeans_{}_{}.pkl".format(i, method))


def kmedoids_clustering(k, enembs, metric, directory, mat):
    for i in k:
        if metric != 'precomputed':
            clustering = KMedoids(n_clusters=i, random_state=0, metric=metric).fit(enembs)
        else:
            clustering = KMedoids(n_clusters=i, random_state=0, metric=metric).fit(mat)
        save_clustering(clustering, "{}/{}_{}.pkl".format(directory, i, metric))



def create_distance_matrix_semantic(embs):
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

if __name__ == "__main__":
    enembs = load_embeddings('../save/DBpedia15k/out_data/pickle/ent_emb.pkl')
    r = [9]
    agglomerative_clustering(r, enembs, 'cosine', 'complete')

    #mat = create_distance_matrix_semantic(enembs)
    #np.save("semantic_matrix", mat)
    '''mat = np.load('semantic_matrix.npy')
    kmedoids_clustering(r, enembs, 'precomputed', '../KMedoids_DBPedia', mat)'''
#python3 explanation/explanation.py --data ./save/DBpedia15k/out_data/pickle/ --save_dir explanation/results/DBpedia15k/euclidian/agglomerative8/2perc/ --multiprocessing True --clustering agglomerative-8
