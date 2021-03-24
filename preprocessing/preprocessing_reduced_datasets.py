import os

import pandas as pd
from multiprocessing import Process, Manager

from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def remove_triples(entities_removed: list, triple_set: pd.DataFrame, out_dict, set_name: str):
    """
    Removes the entities contained in entities_removed from the triple set given in input
    :param set_name: name of  the set processed, useful to save the processed set into the dictionary
    :param out_dict: shared dictionary useful for multiprocessing; needs a Manager.dict() object
    :param entities_removed: list of the entites removed from entites2id, useful to know which are the entities to remove
    from the sets containing these data
    :param triple_set: train, test, or validation set from which to remove triples in order to reduce the set
    :return: an updated triple set
    """

    len_set = len(triple_set)
    for row in range(0, len_set):
        for col in ("head", "tail"):
            entity = triple_set.at[row, col]
            if entity in entities_removed:
                triple_set.iloc[[row]] = None
                break # it is not necessary to check if the entity is present in both head and tail
    out_dict[set_name] = triple_set.dropna()  # dato che non è possibile rimuovere durante il ciclo


def reset_ids(data: pd.DataFrame):
    """
    Reset the ids contained in the dataframe of entities or relations, starting again from 0; reset also the index of
    the dataframe
    :param data: dataframe containing entities or relations mapped with the ids; it has to have the second column named "id"
    :return: updated dataframe
    """
    data.reset_index(drop=True, inplace=True)
    for index, row in data.iterrows():
        data.at[index, "id"] = index  # riorganizzazione degli id

    return data.astype({'id': 'int32'})


def main_reduce_datasets(data_path, dataset_name, fraction_to_remove):
    """
    Reduce a dataset whom files are contained into the path provided; it has to contain entity2id.txt, relation2id.txt,
    train.txt, test.txt, valid.txt
    :param data_path: path of the dataset containing the files listed before
    :param dataset_name: name of the dataset, useful to print messages
    :param fraction_to_remove: an integer for which the len of the entity2id set is divided, in order to remove that amount
    of entities
    :return: nothing
    """
    print(f"\nStarted processing for {dataset_name}")
    entities = pd.read_csv(data_path + 'entity2id.txt', sep='\t', header=None, names=["entity", "id"])
    relations = pd.read_csv(data_path + 'relation2id.txt', sep='\t', header=None, names=["rel", "id"])
    train = pd.read_csv(data_path + 'train.txt', sep='\t', header=None, names=["head", "tail", "rel"])
    test = pd.read_csv(data_path + 'test.txt', sep='\t', header=None, names=["head", "tail", "rel"])
    validation = pd.read_csv(data_path + 'valid.txt', sep='\t', header=None, names=["head", "tail", "rel"])

    num_entities_remove = int(len(entities) / fraction_to_remove)  # numero delle entità che vogliamo rimuovere
    print(f"entites of {dataset_name}: {len(entities)}")
    # memorizziamo le prime x ('num_entities_remove') istanze per poi rimuoverle anche dagli altri set
    removed_entities = []
    for i in range(0, num_entities_remove):
        removed_entities.append(entities.iloc[i][0])  # 0 = prima colonna
    # rimozione delle entità
    entities = entities.iloc[num_entities_remove:]
    # entities = entities.drop([x for x in range(0, num_entities_remove)], inplace=True)
    entities = reset_ids(entities)  # riorganizzazione degli id
    print(f"entites of {dataset_name} after removal: {len(entities)}")
    # print(entities.head(5))
    # multiprocessing
    manager = Manager()

    return_dict = manager.dict()  # dizionario condiviso per ospitare gli output del multiprocessing
    process_list = []
    train_set_proc = Process(target=remove_triples, args=(removed_entities, train, return_dict, "train"))
    test_set_proc = Process(target=remove_triples, args=(removed_entities, test, return_dict, "test"))
    valid_set_proc = Process(target=remove_triples, args=(removed_entities, validation, return_dict, "valid"))
    process_list.append(train_set_proc)
    process_list.append(test_set_proc)
    process_list.append(valid_set_proc)
    # rimozione, dal training set, delle triple contenenti le entità rimosse
    for process in process_list:
        process.start()
    # print done after processes starts to optimize
    len_train = len(train)
    print(f"\ntrain samples of {dataset_name}: {len_train}")
    len_validation = len(validation)
    print(f"validation samples of {dataset_name}: {len_validation}")
    len_test = len(test)
    print(f"test samples of {dataset_name}: {len_test}")
    print(f"Reducing train, valid, and test for {dataset_name}...\n")

    for process in process_list:
        process.join()

    train = return_dict["train"]  # using the shared variable altered by the processes created before
    test = return_dict["test"]
    validation = return_dict["valid"]
    len_train = len(train)
    print(f"train triples of {dataset_name} after removal: {len_train}")

    len_test = len(test)
    print(f"test triples of {dataset_name} after removal: {len_test}")

    len_validation = len(validation)
    print(f"validation triples of {dataset_name} after removal: {len_validation}")

    # rimozione delle relazioni
    print(f"relations of {dataset_name}: {len(relations)} \n Reducing relations of {dataset_name}...")
    for i in range(0, len(relations)):
        candidate_rel = relations.iloc[i]["rel"]
        # controllo che la relazione sia assente da qualsiasi set prima di rimuoverla

        if (candidate_rel not in train.rel.values) and (candidate_rel not in test.rel.values) and (
                candidate_rel not in validation.rel.values):
            relations.iloc[[i]] = None
            # print(f"rimossa la relazione {candidate_rel}")
    relations.dropna(inplace=True)
    relations = reset_ids(relations)  # riorganizzazione degli id
    print(f"relations  of {dataset_name} after removal: {len(relations)}")

    save_dir = f"{data_path}reduced{fraction_to_remove}/" #to save a subfolder with the fraction used
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(save_dir + 'train.txt', sep='\t', header=None, index=None)
    test.to_csv(save_dir + 'test.txt', sep='\t', header=None, index=None)
    validation.to_csv(save_dir + 'valid.txt', sep='\t', header=None, index=None)
    relations.to_csv(save_dir + 'relation2id.txt', sep='\t', header=None, index=None)

    entities.to_csv(save_dir + 'entity2id.txt', sep='\t', header=None, index=None)
    print(f"\nAll files  of {dataset_name} saved into path {save_dir}")


if __name__ == "__main__":
    """
    General idea of the script: removes a portion of the entities from entity2id file, saving them into a list
    in order to remove them also from train, dev, and test sets; then, remove all the relations (from the file relation2id)
    that are not anymore in none of the sets (train, dev, test)
    """
    datasets_dir = 'datasets/'
    """name = "FB15K"
    dir = f"{datasets_dir}{name}/"
    main_reduce_datasets(dir, name, 2)"""
    process_list = []
    for name in os.listdir(datasets_dir):
        dir = f"{datasets_dir}{name}/"
        print(dir)
        process_list.append(Process(target=main_reduce_datasets, args=(dir, name, 1.01)))
        process_list[-1].start() #start the last appended process

    for process in process_list:
        process.join()
    print("All the dataset have been reduced!")
