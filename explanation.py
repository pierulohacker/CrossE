import pickle
import numpy as np

from itertools import islice


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))


class DataManager():

    def __init__(self, pickles_path):
        """
        Initialize the data needed for the explanation process
        :param pickles_path: (has to end with slash e.g C:/pickles/)the path of the folder containing the serialized objects in pickle format
        ent_emb.pkl
        inv_rel_emb.pkl
        rel_emb.pkl
        test_hr_t.pkl
        test_predicted_heads.pkl
        test_predicted_tails.pkl
        test_tr_h.pkl
        test_triples.pkl
        train_hr_t.pkl
        train_tr_h.pkl
        """
        self.__rel_emb = None
        self.__test_hr_t = None
        self.__test_predicted_heads = None
        self.__test_predicted_tails = None
        self.__test_tr_h = None
        self.__test_triples = None
        self.__train_hr_t = None
        self.__train_tr_h = None
        self.__inv_rel = None
        self.__entity_emb = None
        file_names = ["ent_emb.pkl", "inv_rel_emb.pkl", "rel_emb.pkl", "test_hr_t.pkl", "test_predicted_heads.pkl",
                      "test_predicted_tails.pkl", "test_tr_h.pkl", "test_triples.pkl", "train_hr_t.pkl",
                      "train_tr_h.pkl"]

        file_path = pickles_path + file_names[0]
        with open(file_path, 'rb') as f:
            self.__entity_emb = pickle.load(f)
        file_path = pickles_path + file_names[1]
        with open(file_path, 'rb') as f:
            self.__inv_rel = pickle.load(f)

        file_path = pickles_path + file_names[2]
        with open(file_path, 'rb') as f:
            self.__rel_emb = pickle.load(f)

        file_path = pickles_path + file_names[3]
        with open(file_path, 'rb') as f:
            self.__test_hr_t = pickle.load(f)

        file_path = pickles_path + file_names[4]
        with open(file_path, 'rb') as f:
            self.__test_predicted_heads = pickle.load(f)

        file_path = pickles_path + file_names[5]
        with open(file_path, 'rb') as f:
            self.__test_predicted_tails = pickle.load(f)

        file_path = pickles_path + file_names[6]
        with open(file_path, 'rb') as f:
            self.__test_tr_h = pickle.load(f)

        file_path = pickles_path + file_names[7]
        with open(file_path, 'rb') as f:
            self.__test_triples = pickle.load(f)

        file_path = pickles_path + file_names[8]
        with open(file_path, 'rb') as f:
            self.__train_hr_t = pickle.load(f)

        file_path = pickles_path + file_names[9]
        with open(file_path, 'rb') as f:
            self.__train_tr_h = pickle.load(f)

    @property
    def entity_emb(self):
        return self.__entity_emb

    @property
    def inv_rel(self):
        return self.__inv_rel

    @property
    def rel_emb(self):
        return self.__rel_emb

    @property
    def test_hr_t(self):
        return self.__test_hr_t

    @property
    def test_predicted_heads(self):
        return self.__test_predicted_heads

    @property
    def test_predicted_tails(self):
        return self.__test_predicted_tails

    @property
    def test_tr_h(self):
        return self.__test_tr_h

    @property
    def test_triples(self):
        # h,t,r
        return self.__test_triples

    @property
    def train_hr_t(self):
        return self.__train_hr_t

    @property
    def train_tr_h(self):
        return self.__train_tr_h


class Explainer:
    def top_sim_emb(self, emb, emb_id, embedding_matrix, top_k=5):
        """
        Compute the euclidean distance between an embedding of the triple (head, relation, or tail)
        and all the other embeddings of the same kind of objects
        set for wich embeddings are provided
        :param top_k: entities/relationships most similar to return
        :param emb_id: id of the object of the KG, useful to exlude it from the comparison
        :param emb: relationship of the test triple to compare with the other relationships
        :param embedding_matrix: embeddings of the entities/relationships in the KG
        :return: list of ids of the top_k most similar objects to emb
        """
        distances = {}
        for i in range(0, len(embedding_matrix)):
            other_rel = embedding_matrix[i]
            if i != emb_id:
                dst = np.linalg.norm(other_rel - emb)
                distances[i] = dst
        sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        top_k_sorted_dict = take(top_k, sorted_dict.items())
        ids = list(top_k_sorted_dict.keys())
        return ids

    def __tails(self, head, rel, train_dict):
        """
        Explore the train dict to find all the ids of the tails for the triple (h, r, ?) and returns them as a set; if
        there is no triple (h,r,?) returns an empty list
        :param head: id of  the head entity
        :param rel: id of the relationship
        :param train_dict: dictionary to iterate over the triples
        :return: set of ids of the tail entities
        """
        tails = {}
        try:
            tails = train_dict[head][rel]  # tails connected to the head entity via rel, it's a set
        except KeyError as e:
            pass
        return tails

    def __relations(self, head, train_dict):
        """
        Returns all the ids of the relationships connected to the head
        :param head:
        :param train_dict: if hr_t, the method will look for outgoing relationships; if tr_h it will look for ingoing relationships
        :return: set of ids
        """
        relations = {}
        try:
            relations = set(train_dict[head].keys())  # tails connected to the head entity via rel, it's a set
        except KeyError as e:
            pass
        return relations


    def direct_path(self, head, rel, tail, train_dict):
        """
        Check if exists a triple head --rel--> tail
        :param head: head id
        :param rel: relation id
        :param tail: tail id
        :param train_dict: dictionary containing the training triples, useful to explore the KG more efficiently
        :return:
        """
        found = 0
        try:
            tails = train_dict[head][rel]  # tails connected to the head entity via rel, it's a set
            # if the previous does not rise exception, it means that we can have paths of type: 1, 5, 6
            # type 1 path
            if tail in tails:  # controllo che la relazione simile congiunga head e tail
                #print(f"path found: {head} --{rel}--> {tail}")
                found = 1
        except KeyError as e:
            # non esiste nel grafo una tripla che congiunga testa e coda mediante quella relazione
            #print(f"Nessun path: {head} --{rel}--> {tail}")
            found = 0
        return found

    def paths(self, head, rel, tail, train_dicts:list):
        """
        Looks for paths between head and tail via rel
        :param head: id of the head entity
        :param rel: id of the relationship
        :param tail: id of the tail entity
        :param train_dict: dictionary containing the training triples, useful to explore the KG more efficiently
        :return: paths retrieved
        """
        hr_t = train_dicts[0]
        tr_h = train_dicts[1]
        # key = path type; value = occurrences of that type
        paths_expl = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        #print("Type 1:")
        paths_expl[1] = paths_expl[1] + self.direct_path(head, rel, tail, hr_t)


        #print("\nType 2:")
        paths_expl[2] = paths_expl[2] + self.direct_path(tail, rel, head, hr_t)
        ent_to_h = self.__tails(head, rel, tr_h) #NB __tails in questo caso prende tutte le ent che puntano ad h
        ent_from_h = self.__tails(head, rel, hr_t) # entità puntate da h attraverso rel
        rel_ingoing_t = self.__relations(tail, tr_h) # relazioni entranti in tail
        rel_outgoing_t = self.__relations(tail, hr_t) # relazioni uscenti da tail
        #print("\nType 3:")# (h <--rs-- e' --r'--> t)
        for e in ent_to_h:
            # per ogni relazione entrante in t
            for r in rel_ingoing_t:
                paths_expl[3] = paths_expl[3] + self.direct_path(tail, r, e, tr_h)

        #print("\nType 4:")  # (h <--rs-- e' <--r'-- t)
        for e in ent_to_h:
            #per ogni rel uscente da t
            for r in rel_outgoing_t:
                # verifica che t vada in e
                paths_expl[4] = paths_expl[4] + self.direct_path(tail, r, e, hr_t)

        #type 5
        #print("\nType 5:") # (h --rs--> e' --r'--> t)
        for e in ent_from_h: # e' = una qualsiasi entità nel mezzo puntata da h
            # per ogni relazione entrante in t
            for r in rel_ingoing_t:
                paths_expl[5] = paths_expl[5] + self.direct_path(tail, r, e, tr_h)

        #print("\nType 6:") # (h --rs--> e' <--r'-- t)
        for e in ent_from_h:  # e' = una qualsiasi entità nel mezzo puntata da h
            # per ogni relazione uscente da t
            for r in rel_outgoing_t:
                paths_expl[6] = paths_expl[6] + self.direct_path(tail, r, e, hr_t)

        return paths_expl


def main():
    fb15k = DataManager('save/save_FB15k_reduced1.1_2ITER/out_data/pickle/')
    print('loaded')
    print("NB: triple espresse nella forma [h,t,r]\n")
    explainer = Explainer()
    paths_dictionary = {} #{num_tripla: {num_predizione: {paths} } }
    for num_tripla in range(0, len(fb15k.test_triples)):
        tripla_test = fb15k.test_triples[num_tripla]
        print(f"Tripla di test: {tripla_test}")
        # ids
        test_head_id = tripla_test[0]
        test_tail_id = tripla_test[1]
        rel_id = tripla_test[2]
        # embeddings of the test triple
        head_emb = fb15k.entity_emb[test_head_id]
        tail_emb = fb15k.entity_emb[test_tail_id]
        rel_emb = fb15k.rel_emb[rel_id]
        inv_rel_emb = fb15k.inv_rel[rel_id]
        ## TAIL PREDICTION EXPLANATION
        tail_predictions = fb15k.test_predicted_tails[num_tripla]
        # per ogni rel_emb, similarità con il rel_emb della tripla
        sim_rel = explainer.top_sim_emb(rel_emb, rel_id, fb15k.rel_emb, 5)
        #print(f"Relazioni più simili alla relazione {rel_id}:\n{sim_rel}")
        paths_for_pred = {} # dict contenente {num_pred: paths, num_pred1: path1}
        for num_pred in range(0, len(tail_predictions)):
            predicted_tail_id = tail_predictions[num_pred]
            # dunque cercare spiegazione per (head_id, pred_tail, rel_id)

            # ricerca di paths alternativi tra h e t mediante le relazioni simili in sim_rel
            #print(f"Data la predizione {head_id, predicted_tail_id, rel_id}")

            # PATH DI SPIEGAZIONE: ricerca dei path alternativi guidati dalle relazioni simili
            prediction_paths = {}  # conterrà dizionari per i paths che congiungono h e t nella tripla predetta mediante la rel simile ad r
            for sim_id in sim_rel:
                prediction_paths[sim_id] = (explainer.paths(test_head_id, sim_id, predicted_tail_id, [fb15k.train_hr_t, fb15k.train_tr_h]))
            paths_for_pred[num_pred] = prediction_paths
        paths_dictionary[num_tripla] = paths_for_pred

    #print(paths_dictionary)
    print()


if __name__ == '__main__':
    main()
