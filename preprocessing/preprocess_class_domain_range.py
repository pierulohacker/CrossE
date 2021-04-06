"""this file allows to create files containing, for each row, a dictionary with ent_id and the set of classes to which it
belongs; then for relationships domains, rel_id and the set of domain, and the same for ranges"""
import pandas as pd
from explanation.batch_similarity import save_data

def aggregate(df_semantic: pd.DataFrame, ids):
    """
    aggregate for each id, the ids of the semantic informations; e.g.: if you pass ids of the entites and a dataframe containing
    for each row an id of entity and a class, it will aggregate all the classe of that entity into a set
    :param df_semantic: dataframe containing semantic informations: classes for the entities, domains/ranges
    :param ids:
    :return:
    """
    # per ogni entità, raccogliere le classi di appartenenza
    dictionary = {}  # contiene ent_id:{classes} oppure rel_id: {domini} o rel_id: {ranges}
    for one_id in ids:
        dictionary[one_id] = set()
        this_ent_to_sem = df_semantic[
            df_semantic[df_semantic.columns[0]] == one_id]  # prendo solo le righe che contengono quell'id
        if not this_ent_to_sem.empty:
            # print(this_ent_to_class)
            for index, row in this_ent_to_sem.iterrows():
                # prendo le classi della entità
                dictionary[one_id].add(row[df_semantic.columns[1]])
    return dictionary


if __name__ == '__main__':
    dataset_name = 'DBpedia15k'
    dir_path = f"../datasets/{dataset_name}/"
    print(dataset_name)
    entities_to_id = pd.read_csv(f"{dir_path}entity2id.txt", sep='\t', header=None, names=["entity", "id"])
    entities_to_class = pd.read_csv(f"{dir_path}entity2class.txt", sep='\t', header=None,
                                    names=["entity_id", "class_id"])

    relations_to_id = pd.read_csv(f"{dir_path}relation2id.txt", sep='\t', header=None,
                                  names=["relation", "id"])
    """Attenzione, sono separati da spazio e non da tab"""
    relations_to_domain = pd.read_csv(f"{dir_path}rs_domain2id.txt", sep=' ', header=None,
                                      names=["rel_id", "domain_id"])
    relations_to_range = pd.read_csv(f"{dir_path}rs_range2id.txt", sep=' ', header=None,
                                     names=["rel_id", "range_id"])



    ent_ids = set(entities_to_id["id"].values)
    print(f"Num entites: {len(ent_ids)}")

    ent_to_class_dict = aggregate(df_semantic=entities_to_class, ids=ent_ids)
    save_data(ent_to_class_dict, dir_path, "entity2class_dict.pkl")
    print("Classes aggregated and saved for each entity")

    rel_ids = set(relations_to_id["id"].values)
    print("Aggregating domains...")
    rel_to_dom_dict = aggregate(df_semantic=relations_to_domain, ids=rel_ids)
    save_data(rel_to_dom_dict, dir_path, "rs_domain2id_dict.pkl")
    print("Domains aggregated and saved")

    print("Aggregating ranges...")
    rel_to_range_dict = aggregate(df_semantic=relations_to_range, ids=rel_ids)
    save_data(rel_to_range_dict, dir_path, "rs_range2id_dict.pkl")
    print("Ranges aggregated and saved")
