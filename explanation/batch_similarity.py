"""
Compute in batch the similarity between each entity and the similarity between each relation, in order to use these data
to speed up the explanation process
"""
import argparse
import multiprocessing
from pathlib import Path
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm



def load_data(pickle_folder: str):
    """
    load the data needed for the similarity evaluation; the files needed are ent_emb.pkl, rel_emb.pkl, inv_rel.pkl
    :param pickle_folder: folder containing the embeddings
    :return: data loaded in the form of list of lists; E.g ent_emb = [[emb1 numbs], [emb2 numbs], ...]
    """
    file_names = ["ent_emb.pkl", "inv_rel_emb.pkl", "rel_emb.pkl"]
    file_path = f"{pickle_folder}{file_names[0]}"
    with open(file_path, 'rb') as f:
        entity_emb = pickle.load(f)

    file_path = f"{pickle_folder}{file_names[1]}"
    with open(file_path, 'rb') as f:
        rel_emb = pickle.load(f)

    file_path = f"{pickle_folder}{file_names[2]}"
    with open(file_path, 'rb') as f:
        inv_rel_emb = pickle.load(f)

    return entity_emb, rel_emb, inv_rel_emb

def __euclidian(a,b):
    """
    Computes euclidian distance between two lists
    :return: distance
    """
    return np.linalg.norm(a - b)

def __cosine(a,b):
    """
    Computes the cosine similarity between a and b
    :param a:
    :param b:
    :return:
    """
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def __top_sim_emb(emb, emb_id, embedding_matrix, distance_type, first_n=15):
    """
    Compute the euclidean distance between an embedding of the triple (head, relation, or tail)
    and all the other embeddings of the same kind of objects set for wich embeddings are provided
    :param top_k: entities/relationships most similar to return
    :param emb_id: id of the object of the KG, useful to exlude it from the comparison
    :param emb: relationship of the test triple to compare with the other relationships
    :param embedding_matrix: embeddings of the entities/relationships in the KG
    :param first_n: number of top similar embeddings ids to keep, it helps to reduce the size required to store files; also
    the explnation experiments does not require all the ids, but only the first (at most 15) will be used
    :return: list of ids of the top_k most similar objects to emb
    """
    if distance_type == 'euclidian':
        distance_function = __euclidian
    elif distance_type == 'cosine':
        distance_function = __cosine
    distances = {}  # dizionario {id: distanza dall'emb, id: distanza,...}
    for i in range(0, len(embedding_matrix)):
        if i != emb_id:
            other_rel = embedding_matrix[i]
            dst = distance_function(other_rel, emb)
            distances[i] = dst
    sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    ids = list(sorted_dict.keys())
    return ids[:first_n]


def compute_sim_dictionary(embeddings: list, dict_multiproc, proc_name, distance_type):
    """
    Compute the similarity between each element in the embedding list
    :param embeddings: embeddings list in the form [[emb1], [emb2], ...]
    :return: dictionary in the form {emb_id: [sim_emb_id1, sim_emb_id2]} ranked by similarity
    """
    similarity_dictionary = {}
    for emb_id in range(0, len(embeddings)):
        similarity_dictionary[emb_id] = __top_sim_emb(embeddings[emb_id], emb_id, embeddings, distance_type)
    dict_multiproc[proc_name] = similarity_dictionary


def save_data(sim_dict, save_path, filename):
    """
    Saves the dictionary containing emb ids and the lists of similar embeddings ids
    :param sim_dict:
    :param save_path:
    :return:
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = f"{save_path}{filename}"
    with open(file_path, 'wb') as f:
        pickle.dump(sim_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch similarity.')
    parser.add_argument('--data', dest='data_dir', type=str,
                        help="Data folder containing the output of the training phase (pickle_files")

    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        help='directory to save in the similarity data; default will be set to the same folder the data'
                             'are loaded from.',
                        default='data_dir')
    parser.add_argument('--distance', dest='distance_type', type=str,
                        help='choose the distance to compute between entities, possible choises: euclidian, cosine',
                        default='euclidian')
    global args
    args = parser.parse_args()

    data_folder = args.data_dir

    if args.distance_type != "euclidian" and args.distance_type != "cosine":
        print(f"Distance type: {args.distance_type} is not a valid value")
        exit()
    print(f"Distance type: {args.distance_type}")
    save_dir = args.save_dir
    if save_dir == 'data_dir':
        save_dir = f"{data_folder}{args.distance_type}/"
    ent, rel, inv = load_data(data_folder)
    manager1 = multiprocessing.Manager()
    return_dict = manager1.dict()

    processes_list = []
    print("Computing similarity between entities")
    p1 = multiprocessing.Process(target=compute_sim_dictionary, args=(ent, return_dict, "ent", args.distance_type))
    processes_list.append(p1)
    p1.start()
    print("Computing similarity between relationships")
    p2 = multiprocessing.Process(target=compute_sim_dictionary, args=(rel, return_dict, "rel", args.distance_type))
    processes_list.append(p2)
    p2.start()
    print("Computing similarity between inverse relationships")
    p3 = multiprocessing.Process(target=compute_sim_dictionary, args=(inv, return_dict, "inv", args.distance_type))
    processes_list.append(p3)
    p3.start()

    for proc in processes_list:
        proc.join()

    sim_ent = return_dict['ent']
    sim_rel = return_dict['rel']
    sim_inv_rel = return_dict['inv']


    save_data(sim_ent, save_path=f"{save_dir}/", filename="sim_entities.pkl")
    save_data(sim_rel, save_path=f"{save_dir}/", filename="sim_rel.pkl")
    save_data(sim_inv_rel, save_path=f"{save_dir}/", filename="sim_inv_rel.pkl")
    print(f"All data  stored in {save_dir}")

    print()
