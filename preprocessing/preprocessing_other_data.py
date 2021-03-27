"""
This script is designed to process the datasets proposed here --> https://github.com/Keehl-Mihael/TransROWL-HRS
"""
import argparse

import pandas as pd
from varname import nameof

def id_to_verbose(dataset: pd.DataFrame, mapping_ent: pd.DataFrame, mapping_rel: pd.DataFrame):
    """
    Convert a dataset basd on ids to a dataset based on verbose names for entities and relationships
    :param dataset: dataset to convert
    :param mapping_ent: file that maps  entites to id
    :param mapping_rel: file that maps  relations to id
    :return: dataset with verbose name instead of ids
    """
    # heads
    #print(f"Dataset before merge:\n{dataset}")
    merged = dataset.merge(mapping_ent, left_on='head', right_on='id')
    #print(merged)
    merged = merged[['entity', 'tail', 'rel']]
    merged.rename(columns={'entity': 'head'}, inplace=True)
    #merged.columns = ['head', 'tail', 'rel']
    #print(f"Merged heads:\n{merged}")
    # tails
    merged = merged.merge(mapping_ent, left_on='tail', right_on='id')
    merged = merged[['head', 'entity', 'rel']]
    merged.rename(columns={'entity': 'tail'}, inplace=True)
    print(f"Merged tails:\n{merged.values}")
    #merged = merged[['entity', 'tail', 'rel']]
    #merged.columns = ['head', 'tail', 'rel']

    # relationships
    merged = merged.merge(mapping_rel, left_on='rel', right_on='id')
    merged = merged[['head', 'tail', 'rel_y']]
    merged.rename(columns={'rel_y': 'rel'}, inplace=True)
    #print(f"Merged rels:\n{merged.columns}")
    #print(f"Merged rels:\n{merged.values}")
    return merged


    #print(merged)

def main_preprocess(data_path):
    """
    Preprocess the dataset in order to obtain the correct mapping required by CrossE model, more informations in the
    docs
    :param data_path: path of the dataset containing the files listed before
    :return: nothing
    """
    print(f"\nStarted processing...")
    # ignorare la prima riga (contenente il numero di istanze) per i files entity2id.txt relation2id.txt valid2id.txt test2id.txt train2id.txt
    entities2id = pd.read_csv(data_path + 'entity2id.txt', sep='\t', header=None, names=["entity", "id"], skiprows=1)
    relations2id = pd.read_csv(data_path + 'relation2id.txt', sep='\t', header=None, names=["rel", "id"], skiprows=1)
    train2id = pd.read_csv(data_path + 'train2id.txt', sep=' ', header=None, names=["head", "tail", "rel"], skiprows=1)
    test2id = pd.read_csv(data_path + 'test2id.txt', sep=' ', header=None, names=["head", "tail", "rel"], skiprows=1)
    validation2id = pd.read_csv(data_path + 'valid2id.txt', sep=' ', header=None, names=["head", "tail", "rel"], skiprows=1)
    triples = pd.read_csv(data_path + 'triples.txt', sep=' ', header=None, names=["head", "rel", "tail"])

    print(f"entities2id: \n {entities2id.head(5)}")
    print(f"relations2id: \n {relations2id.head(5)}")
    print(f"train2id: \n {train2id.head(5)}")
    print(f"test2id: \n {test2id.head(5)}")
    print(f"validation2id: \n {validation2id.head(5)}")
    print(f"triples: \n {triples.head(5)}")

    # re-ordering the columns
    columns_in_order = ["head", "tail", "rel"]

    triples = triples.reindex(columns=columns_in_order)
    print("AFTER REINDEXING OF THE COLUMNS (order now is head,tail,relation)")

    print(f"triples: \n {triples.head(5)}")

    # generation of train.txt, test.txt, valid.txt not bades on ids but on verbose names
    train_verbose = id_to_verbose(train2id, entities2id, relations2id)
    test_verbose = id_to_verbose(test2id, entities2id, relations2id)
    validation_verbose = id_to_verbose(validation2id, entities2id, relations2id)
    print(f"Verbose training set: \n{train_verbose}\n")
    print(f"Verbose test set: \n{test_verbose}")
    print(f"Verbose validation set: \n{validation_verbose}")
    return entities2id, relations2id, train_verbose, validation_verbose, test_verbose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing external dataset --> https://github.com/Keehl-Mihael/TransROWL-HRS')
    parser.add_argument('--data', dest='data_dir', type=str,
                        help="Data folder containing the data")

    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        help='directory to save in the processed data, useful to fit into CrossE',
                        default='data_dir')

    args = parser.parse_args()
    #load_folder = "../datasets/DBpedia15k/original_files/"
    load_folder = args.data_dir
    entities2id, relations2id, train_verbose, validation_verbose, test_verbose = main_preprocess(load_folder)
    #save_folder = "../datasets/DBpedia15k/"
    save_folder = args.save_dir
    # save
    entities2id.to_csv(save_folder + 'entity2id.txt', sep='\t', header=None, index=None)
    relations2id.to_csv(save_folder + 'relation2id.txt', sep='\t', header=None, index=None)
    train_verbose.to_csv(save_folder + 'train.txt', sep='\t', header=None, index=None)
    validation_verbose.to_csv(save_folder + 'valid.txt', sep='\t', header=None, index=None)
    test_verbose.to_csv(save_folder + 'test.txt', sep='\t', header=None, index=None)
    print(f"All data saved into {save_folder}")
