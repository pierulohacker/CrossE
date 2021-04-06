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

class semantic_data:
    """
    Static objects class to host semantic data
    """
    entity2class_dict = {}
    rs_domain2id_dict = {}
    rs_range2id_dict = {}


def load_data(pickle_folder: str):
    """
    load the data needed for the similarity evaluation; the files needed are ent_emb.pkl, rel_emb.pkl, inv_rel.pkl
    :param pickle_folder: folder containing the embeddings
    :return: data loaded in the form of list of lists; E.g ent_emb = [[emb1 numbs], [emb2 numbs], ...]
    """
    file_names = ["ent_emb.pkl", "inv_rel_emb.pkl", "rel_emb.pkl", "entity2class_dict.pkl", "rs_domain2id_dict.pkl", "rs_range2id_dict.pkl"]
    file_path = f"{pickle_folder}{file_names[0]}"
    with open(file_path, 'rb') as f:
        entity_emb = pickle.load(f)

    file_path = f"{pickle_folder}{file_names[1]}"
    with open(file_path, 'rb') as f:
        rel_emb = pickle.load(f)

    file_path = f"{pickle_folder}{file_names[2]}"
    with open(file_path, 'rb') as f:
        inv_rel_emb = pickle.load(f)

    entity2class_dict = None
    rs_domain2id_dict = None
    rs_range2id_dict = None
    if args.distance_type == 'semantic':
        # entità e classi
        file_path = f"{args.semantic_dir}{file_names[3]}"
        with open(file_path, 'rb') as f:
            entity2class_dict = pickle.load(f)
        # relazioni e domini
        file_path = f"{args.semantic_dir}{file_names[4]}"
        with open(file_path, 'rb') as f:
            rs_domain2id_dict = pickle.load(f)
        # relazioni e ranges
        file_path = f"{args.semantic_dir}{file_names[5]}"
        with open(file_path, 'rb') as f:
            rs_range2id_dict = pickle.load(f)


    return entity_emb, rel_emb, inv_rel_emb, entity2class_dict, rs_domain2id_dict, rs_range2id_dict


def __euclidean(a, b, id_a=None, id_b=None, obj_type=None, classes=None, domains=None, ranges=None):
    """
    Computes euclidian distance between two lists; the other paramas are not used, is only useful to simplify the code for semantic evaluations
    :return: distance
    """
    return np.linalg.norm(a - b)


def __cosine(a, b, id_a=None, id_b=None, obj_type=None, classes=None, domains=None, ranges=None):
    """
    Computes the cosine similarity between a and b; the other paramas are not used, is only useful to simplify the code for semantic evaluations
    :param a: embedding a
    :param b: embedding b
    :return: cosine similarity between two embeddings a and b
    """
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def __jaccard(set1, set2):
    """
    Computes jaccard index, given two sets
    :param set1:
    :param set2:
    :return: similarity between the two sets in terms of jaccard index
    """
    numeratore = len(set1.intersection(set2))
    denominatore = len(set1.union(set2))  # se è pari a 0 significa che nessuno dei due ha nulla
    if denominatore == 0:
        #todo: eventualmente restituiremo -1 o altro per denotare che bisogna intervenire con un'altra misura di similarità
        return 0  # arbitrariamente per denotare che entrambi sono privi di qualsiasi info
    else:
        return numeratore / denominatore


def __semantic(emb_a, emb_b, id, other_id, obj_type, classes, domains, ranges):
    """
    Compute a semantic distance based on jaccard index for entities, based on the classes,
     and an abstraction of jaccard index for relationships, based on domain and range; if these semantic informations are
     not available, it will compute the euclidian distance
    :param emb_a: embedding of the first entity/relation corresponding to 'id'
    :param emb_b: embedding of the second entity/relation corresponding to 'id'
    :param id: id of the entity/relationship
    :param other_id: id of the other entity/relatioship
    :return: similarity value
    """
    # if the object type is 'ent' we have to compute a simple jaccard index and euclidian distance if there is no domains
    if obj_type == 'ent':
        # prendiamo i dati delle entità riguardanti le classi
        return __jaccard(classes[id], classes[other_id])


    elif obj_type == 'rel':
        dom_jaccard = __jaccard(domains[id], domains[other_id])
        range_jaccard = __jaccard(ranges[id], ranges[other_id])
        #todo: eventualmente qui andremo a mediare dividendo per 2, altrimenti al momento abbiamo [0;2]
        return dom_jaccard + range_jaccard
    else:
        raise ValueError(f"obj_type can be either 'ent' or 'rel', not '{obj_type}'")



def __top_sim_emb(emb, emb_id, embedding_matrix, distance_type, obj_type, classes=None, domains=None, ranges=None, first_n=15):
    """
    Compute the distance/similarity between an embedding of the triple (head, relation, or tail)
    and all the other embeddings of the same kind of objects set for wich embeddings are provided; if the mode is semantic,
    the similarity will be a mix between semantic informations and euclidian distances
    :param top_k: entities/relationships most similar to return
    :param emb_id: id of the object of the KG, useful to exlude it from the comparison
    :param emb: relationship of the test triple to compare with the other relationships
    :param embedding_matrix: embeddings of the entities/relationships in the KG
    :param obj_type: either 'rel' or 'ent', useful only for the semantic distance
    :param first_n: number of top similar embeddings ids to keep, it helps to reduce the size required to store files; also
    the explnation experiments does not require all the ids, but only the first (at most 15) will be used
    :return: list of ids of the top_k most similar objects to emb
    """
    if distance_type == 'euclidian':
        distance_function = __euclidean
    elif distance_type == 'cosine':
        distance_function = __cosine
    elif distance_type == 'semantic':
        distance_function = __semantic

    distances = {}  # dizionario {id: distanza dall'emb, id: distanza,...}
    for i in range(0, len(embedding_matrix)):
        if i != emb_id:
            other_rel = embedding_matrix[i]
            dst = distance_function(other_rel, emb, emb_id, i, obj_type, classes, domains, ranges)
            distances[i] = dst


    if distance_type == 'cosine' or distance_type == 'semantic':
        sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1] , reverse=True)}  # descending
    else: #euclidean
        sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}  # ascending
    ids = list(sorted_dict.keys())
    return ids[:first_n]


def compute_sim_dictionary(embeddings: list, dict_multiproc, proc_name, distance_type, obj_type, classes=None, domains=None, ranges=None):
    """
    Compute the similarity between each element in the embedding list
    :param obj_type: either 'rel' or 'ent', useful only for the semantic distance
    :param embeddings: embeddings list in the form [[emb1], [emb2], ...]
    :return: dictionary in the form {emb_id: [sim_emb_id1, sim_emb_id2]} ranked by similarity
    """
    similarity_dictionary = {}
    for emb_id in range(0, len(embeddings)):
        similarity_dictionary[emb_id] = __top_sim_emb(embeddings[emb_id], emb_id, embeddings, distance_type, obj_type, classes, domains, ranges)
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
                        help='choose the distance to compute between entities, possible choises: euclidian, cosine, semantic',
                        default='euclidian')
    parser.add_argument('--semantic_data', dest='semantic_dir', type=str,
                        help="Data folder containing the dictionaries about relation domains and ranges, and classes"
                             "for the entities; tipically it's the folder in which the raw dataset files are stored."
                             "By default it is None, because by default the similarity function is the euclidean distance.",
                        default=None)


    global args
    args = parser.parse_args()

    data_folder = args.data_dir

    if args.distance_type != "euclidian" and args.distance_type != "cosine" and args.distance_type != "semantic":
        print(f"Distance type: {args.distance_type} is not a valid value")
        exit()

    # più comodo per specificare gli args
    if args.semantic_dir != None:
        args.distance_type = 'semantic'
    save_dir = args.save_dir
    if save_dir == 'data_dir':
        save_dir = f"{data_folder}{args.distance_type}/"


    if args.distance_type == 'semantic' and args.semantic_dir is None:
        print("You had to provide a folder (--semantic_data) in which there are three files: entity2class_dict.pkl, "
              "rs_domain2id_dict.pkl, rs_range2id_dict.pkl. \nEXIT")
        exit()
    print(f"Distance type: {args.distance_type}")
    print(f"Save folder: {save_dir}")
    ent, rel, inv, classes, domains, ranges = load_data(data_folder) # classe, domains, ranges will be None if not semantic mode
    semantic_data.entity2class_dict = classes

    manager1 = multiprocessing.Manager()
    return_dict = manager1.dict()

    processes_list = []
    print("Computing similarity between entities")
    p1 = multiprocessing.Process(target=compute_sim_dictionary, args=(ent, return_dict, "ent", args.distance_type, 'ent', classes, domains, ranges))
    processes_list.append(p1)
    p1.start()
    print("Computing similarity between relationships")
    p2 = multiprocessing.Process(target=compute_sim_dictionary, args=(rel, return_dict, "rel", args.distance_type, 'rel', classes, domains, ranges))
    processes_list.append(p2)
    p2.start()
    if args.distance_type != 'semantic':
        print("Computing similarity between inverse relationships")
        p3 = multiprocessing.Process(target=compute_sim_dictionary, args=(inv, return_dict, "inv", args.distance_type, 'rel'))
        processes_list.append(p3)
        p3.start()

    for proc in processes_list:
        proc.join()

    sim_ent = return_dict['ent']
    sim_rel = return_dict['rel']
    if args.distance_type != 'semantic':
        sim_inv_rel = return_dict['inv']
    print("Data computed, start saving...")
    save_data(sim_ent, save_path=f"{save_dir}/", filename="sim_entities.pkl")
    save_data(sim_rel, save_path=f"{save_dir}/", filename="sim_rel.pkl")
    if args.distance_type != 'semantic':
        save_data(sim_inv_rel, save_path=f"{save_dir}/", filename="sim_inv_rel.pkl")
    print(f"All data  stored in {save_dir}")

    print()
