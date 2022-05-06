import pickle
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def boh(k, method):
    with open('../save/FB15k-237-swapped/out_data/pickle/ent_emb.pkl', 'rb') as f:
        enembs = pickle.load(f)
    for i in k:
        clustering = KMeans(n_clusters=i, algorithm=method).fit(enembs)
        with open("KMeans_{}_{}.pkl".format(i, method), 'wb') as f:
            pickle.dump(clustering, f)

'''plt.plot(k, distorsion, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
'''

if __name__ == "__main__":
    r = [8,10,15,20]
    boh(r, "auto")
