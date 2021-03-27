"""
CrossE uses datasets in which the triples are in the form "subject  object  relation"; use this script if you want to
alterate datasets that are in the form "subject  relation    object"
"""
import pandas as pd


def load_dataset(data_path) -> pd.DataFrame():
    """
    Load the data contained in the dir_path
    :type data_path: str
    :param data_path: directory of the dataset,containing entity2id.txt, relation2id.txt,
    train.txt, test.txt, valid.txt
    :return: dataframes for the data contained in the files dir_path/train.txt, dir_path/test.txt, dir_path/valid.txt
    """
    train = pd.read_csv(data_path + 'train.txt', sep='\t', header=None, names=["head", "rel", "tail"])
    test = pd.read_csv(data_path + 'test.txt', sep='\t', header=None, names=["head", "rel", "tail"])
    validation = pd.read_csv(data_path + 'valid.txt', sep='\t', header=None, names=["head", "rel", "tail"])
    return train, test, validation


if __name__ == "__main__":
    #it will save in the same folder, so copy the original dataset as backup
    path = "../datasets/FB15k-237-swapped/"
    train, test, valid = load_dataset(path)
    columns_in_order = ["head", "tail", "rel"]
    train = train.reindex(columns=columns_in_order)
    test = test.reindex(columns=columns_in_order)
    valid = valid.reindex(columns=columns_in_order)

    train.to_csv(path + 'train.txt', sep='\t', header=None, index=None)
    test.to_csv(path + 'test.txt', sep='\t', header=None, index=None)
    valid.to_csv(path + 'valid.txt', sep='\t', header=None, index=None)

    print(f"\nAll files saved into path {path}")
