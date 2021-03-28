"""
Compute in batch the similarity between each entity and the similarity between each relation, in order to use these data
to speed up the explanation process
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
from global_logger import Log


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


def __top_sim_emb(emb, emb_id, embedding_matrix):
    """
    Compute the euclidean distance between an embedding of the triple (head, relation, or tail)
    and all the other embeddings of the same kind of objects set for wich embeddings are provided
    :param top_k: entities/relationships most similar to return
    :param emb_id: id of the object of the KG, useful to exlude it from the comparison
    :param emb: relationship of the test triple to compare with the other relationships
    :param embedding_matrix: embeddings of the entities/relationships in the KG
    :return: list of ids of the top_k most similar objects to emb
    """
    distances = {}  # dizionario {id: distanza dall'emb, id: distanza,...}
    for i in range(0, len(embedding_matrix)):
        other_rel = embedding_matrix[i]
        if i != emb_id:
            dst = np.linalg.norm(other_rel - emb)
            distances[i] = dst
    sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    ids = list(sorted_dict.keys())
    return ids


def compute_sim_dictionary(embeddings: list):
    """
    Compute the similarity between each element in the embedding list
    :param embeddings: embeddings list in the form [[emb1], [emb2], ...]
    :return: dictionary in the form {emb_id: [sim_emb_id1, sim_emb_id2]} ranked by similarity
    """
    similarity_dictionary = {}
    for emb_id in range(0, len(embeddings)):
        similarity_dictionary[emb_id] = __top_sim_emb(embeddings[emb_id], emb_id, embeddings)
    return similarity_dictionary


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
    global args
    args = parser.parse_args()

    data_folder = args.data_dir
    save_dir = args.save_dir
    if save_dir == 'data_dir':
        save_dir = data_folder
    ent, rel, inv = load_data(data_folder)
    print("Computing similarity between entities")
    sim_ent = compute_sim_dictionary(ent)
    print("Computing similarity between relationships")
    sim_rel = compute_sim_dictionary(rel)
    print("Computing similarity between inverse relationships")
    sim_inv_rel = compute_sim_dictionary(inv)

    save_data(sim_ent, save_path=f"{data_folder}/", filename="sim_entities.pkl")
    save_data(sim_rel, save_path=f"{data_folder}/", filename="sim_rel.pkl")
    save_data(sim_inv_rel, save_path=f"{data_folder}/", filename="sim_inv_rel.pkl")

    print()
