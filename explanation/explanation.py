import argparse
import pickle
import time
from pathlib import Path

from global_logger import Log
import multiprocessing
from itertools import islice

from threading import Thread, RLock




def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))


class DataManager():

    @staticmethod
    def reduce_predictions(list_of_pred, percentage):
        """
        Reduce the list of arrays o predictions given in input maintaing only a specified percentage of predictions
        :param list_of_pred: list of arrays; each element of the list correspond to the predictions obtained for the test
        triple corresponding to the list id
        :param percentage: percentage of predictions to maintain
        :return: reduced data; the list is the same size because the test triples corresponding are not reduced, but the predictions
        (so the arrays in it) are reduced
        """
        reduced_predictions = []
        # riduzione
        limit_to = len(list_of_pred[0]) * (percentage / 100)
        limit_to = round(limit_to)  # arrotondamento
        for p in list_of_pred:
            reduced_predictions.append(p[:limit_to])
        return reduced_predictions

    def __init__(self, pickles_path, percentage_predictions):
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
        self.__relation_id_sim = None  # dict containing, for each id, the list of most similar embeddings ids
        self.__test_hr_t = None
        self.__test_predicted_heads = None
        self.__test_predicted_tails = None
        self.__test_tr_h = None
        self.__test_triples = None
        self.__train_hr_t = None
        self.__train_tr_h = None
        self.__inv_rel = None
        self.__inv_rel_id_sim = None  # dict containing, for each id, the list of most similar embeddings ids
        self.__entity_emb = None
        self.__entity_id_sim = None  # dict containing, for each id, the list of most similar embeddings ids
        file_names = ["ent_emb.pkl", "inv_rel_emb.pkl", "rel_emb.pkl", "test_hr_t.pkl", "test_predicted_heads.pkl",
                      "test_predicted_tails.pkl", "test_tr_h.pkl", "test_triples.pkl", "train_hr_t.pkl",
                      "train_tr_h.pkl", "sim_entities.pkl", "sim_rel.pkl", "sim_inv_rel.pkl"]

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
        #reduction of predictions
        self.__test_predicted_heads = DataManager.reduce_predictions(self.__test_predicted_heads, percentage_predictions)

        file_path = pickles_path + file_names[5]
        with open(file_path, 'rb') as f:
            self.__test_predicted_tails = pickle.load(f)
        # reduction of tail predictions
        self.__test_predicted_tails = DataManager.reduce_predictions(self.__test_predicted_tails,
                                                                     percentage_predictions)

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

        similarity_data_path = f"{pickles_path}{args.distance_type}/"
        file_path = similarity_data_path + file_names[10]
        with open(file_path, 'rb') as f:
            self.__entity_id_sim = pickle.load(f)

        file_path = similarity_data_path + file_names[11]
        with open(file_path, 'rb') as f:
            self.__relation_id_sim = pickle.load(f)

        file_path = similarity_data_path + file_names[12]
        with open(file_path, 'rb') as f:
            self.__inv_rel_id_sim = pickle.load(f)

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

    @property
    def entities_similarities_dict(self):
        return self.__entity_id_sim

    @property
    def relations_similarities_dict(self):
        return self.__relation_id_sim

    @property
    def inv_relations_similarities_dict(self):
        return self.__inv_rel_id_sim


class Explainer:
    class Explanation:
        """
        Classe innestata, adibita alla creazione di oggetti spiegazione: un oggetto spiegazione corrisponde a una tripla (h,r,t)
        o un path (h,r,e,r',t) e una serie di triple che danno supporto ad essa
        """

        def __init__(self, path, support_paths=None):
            """
            :param path: path found for the explanation
            :param support_paths: list of paths that support the explanation path
            :type support_paths: list of lists
            :type path: list
            """

            if support_paths is None:
                support_paths = []
            self.__path = path
            self.__support_paths = support_paths

        @property
        def path(self):
            return self.__path

        @property
        def support_paths(self):
            return self.__support_paths

        def add_support_path(self, sup_triple: list, similar_path: list):
            """
            Add a support triple
            :type similar_path: list of couples of lists
            :param similar_path: a sequence of head, relation, tail which denotes a path between the head and the tail of
            the triple via another path; the sequence will be at most of length h,r,e1,r1,t
            :param sup_triple: triple that support the explanation path contained in self.__path
            :return:
            """
            self.__support_paths.append([sup_triple, similar_path])

        def __str__(self):
            """Stampa un oggetto explanation come spiegazione - esempio simile - supporto"""
            stringa = ""
            stringa += (f"\tExplanation: {self.__path}")
            stringa += (f"\n\t\tSupports: ")
            for sup in self.__support_paths:
                stringa += (f"\n\t\t\t example: {sup[0]} \t support for this example: {sup[1]}")

            return stringa

    def top_sim_emb(self, emb_id, similarity_dict, top_k=5):
        """
        Extract the top_k similar embeddings id for the emb_id provided; it uses the data contained in the batch-evaluated
        distances between each embedding output of the training phase
        :param similarity_dict: dictionary of the similarities computed, in the form {id: [sim_emb_id1, sim_emb_id2,...], id2: [...],...}
        :param top_k: entities/relationships most similar to return
        :param emb_id: id of the object of the KG, useful to exlude it from the comparison
        :return: list of ids of the top_k most similar objects to emb
        """
        list_all_similaties = similarity_dict[emb_id]

        return list_all_similaties[:top_k]

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
        :return: boolean, True if path has been found
        """
        found = False
        try:
            tails = train_dict[head][rel]  # tails connected to the head entity via rel, it's a set
            # if the previous does not rise exception, it means that we can have paths of type: 1, 5, 6
            # type 1 path
            if tail in tails:  # controllo che la relazione simile congiunga head e tail
                found = True
        except KeyError as e:
            # non esiste nel grafo una tripla che congiunga testa e coda mediante quella relazione
            found = False
        return found

    def paths(self, head, relationship, sim_relationships, tail, train_dicts: list, similar_heads, similar_tails):
        """
        Looks for paths between head and tail via relationships provided in input; there are 6 paths that can be detected
        :param return_dict: dictionary to store data using multiprocessing
        :param similar_tails:
        :param similar_heads:
        :param head_emb:
        :param tail_emb:
        :param emb_matrix:
        :param relationship: id of the relationship of the triple
        :param sim_relationships: list of similar relationships relatioships for which to find a path
        :param head: id of the head entity
        :param tail: id of the tail entity
        :param train_dicts: list of 2 dictionaries containing the training triples from head and tail perspective, respectively;
         useful to explore the KG more efficiently
        :return: paths retrieved for each type with support from other triples and paths into the training graph
        """
        hr_t = train_dicts[0]
        tr_h = train_dicts[1]
        # key = path type; value = occurrences of that type
        paths_expl = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        # looking for the similar entities to the head and to the tail
        top_k = 10  # top similar entities to retrieve
        """
        paths_expl conterrà le spiegazioni di 6 tipologie diverse per ogni predizione della tripla passata in input;
        per il tipo 1 conterrà una lista di triple 1:[[h,rs,t]] e analogamente anche il tipo 2
        tipo 3 avremo 3:[[h,rs,e',r',t], [h,rs,e',r',t],...] e saranno lette come h <--rs-- e' --r'--> t
        etc.
        Sarà il modo in cui andremo a leggere i dati a cambiare per ogni tipologia
        """
        lock = RLock()  # to manage resources in the multithread path finder
        thread_list = []
        for sim_rel in sim_relationships:
            t = Thread(target=self.__multithread_path_finder,
                       args=(head, sim_rel, tail, hr_t, similar_heads, similar_tails, relationship,
                             paths_expl, lock, tr_h))
            t.start()
            thread_list.append(t)
            # multithread_path_find
        for t in thread_list:
            t.join()

        none_counter = 0  # useful to assign None to paths_expl when there is not any explaination
        for expl_type_paths in paths_expl.values():
            if not expl_type_paths:  # se è vuota
                none_counter += 1
        if none_counter == 6:
            paths_expl = {None}

        return paths_expl

    def __multithread_path_finder(self, head, sim_rel, tail, hr_t, similar_heads, similar_tails, relationship,
                                  paths_expl, lock: RLock, tr_h):
        """
        Function executed in multithread; compute all the paths based on 6 types
        :param head: head of the predicted triple for which to find an explanation
        :param sim_rel: similar relation to the
        :param tail: tail of the predicted triple for which to find an explanation
        :param hr_t: dictionary mapping heads and relations to tails
        :param similar_heads: entities similar to head
        :param similar_tails: entities similar to tail
        :param relationship: relationship of the predicted triple for which to find an explanation
        :param paths_expl: dictionary that will contain the explanations of each type for the predicted triple for which
        we are searching an explanation
        :param lock: lock the dictionary paths_expl when it is necessary to save an explanation
        :param tr_h: dictionary mapping tails and relations to heads
        :return:
        """

        # TYPE 1
        if self.direct_path(head, sim_rel, tail, hr_t):
            expl = self.Explanation([head, sim_rel, tail])
            # FIND SUPPORT
            for sim_h in similar_heads:
                for sim_t in similar_tails:
                    # controllare collegamento diretto tra le due entità simili attraverso la rel. originale +
                    if self.direct_path(sim_h, relationship, sim_t, hr_t) and self.direct_path(sim_h, sim_rel,
                                                                                               sim_t, hr_t):
                        expl.add_support_path([sim_h, relationship, sim_t], [sim_h, sim_rel, sim_t])
            if expl.support_paths:  # solo se si è trovato almeno un supporto si andrà a salvare la spiegazione
                with lock:
                    paths_expl[1].append(expl)

        # ("\nType 2:")
        if self.direct_path(tail, sim_rel, head, hr_t):
            expl = self.Explanation([tail, sim_rel, head])
            # FIND SUPPORT
            for sim_h in similar_heads:
                for sim_t in similar_tails:
                    # controllare collegamento diretto tra le due entità simili attraverso la rel. originale + tra loro con rel simile
                    if self.direct_path(sim_h, relationship, sim_t, hr_t) and self.direct_path(sim_t, sim_rel,
                                                                                               sim_h, hr_t):
                        expl.add_support_path([sim_h, relationship, sim_t], [sim_t, sim_rel, sim_h])

            if expl.support_paths:
                with lock:
                    paths_expl[2].append(expl)

        ent_to_h = self.__tails(head, sim_rel,
                                tr_h)  # NB __tails in questo caso prende tutte le ent che puntano ad h

        ent_from_h = self.__tails(head, sim_rel, hr_t)  # entità puntate da h attraverso rel
        rel_ingoing_t = self.__relations(tail, tr_h)  # relazioni entranti in tail
        rel_outgoing_t = self.__relations(tail, hr_t)  # relazioni uscenti da tail

        # Type 3 (h <--rs-- e' --r'--> t)
        for e in ent_to_h:
            # top 10 similar to e and connected to hs (intersezione tra le es e le uscenti da hs
            # ent_sim_e_to_hs = set(self.top_sim_emb(emb_matrix[e], e, emb_matrix, top_k)).intersection()
            for r in rel_ingoing_t:  # per ogni relazione entrante in t
                if self.direct_path(tail, r, e, tr_h):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_to_hs = self.__tails(sim_h, sim_rel, tr_h)  # entità che vanno in hs
                        for sim_t in similar_tails:
                            for sim_e in e_to_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) and self.direct_path(sim_e,
                                                                                                           sim_rel,
                                                                                                           sim_h,
                                                                                                           hr_t) and self.direct_path(
                                    sim_e, r, sim_t, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])

                    if expl.support_paths:  # solo se si è trovato almeno un supporto si andrà a salvare la spiegazione
                        with lock:
                            paths_expl[3].append(expl)
            # (h <--rs-- e' <--r'-- t) tipo 4 insieme al 3 dato che il primo for è uguale
            for r in rel_outgoing_t:  # per ogni rel uscente da t
                # verifica che t vada in e
                # paths_expl[4] = paths_expl[4] + self.direct_path(tail, r, e, hr_t)
                if self.direct_path(tail, r, e, hr_t):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_to_hs = self.__tails(sim_h, sim_rel, tr_h)  # entità che vanno in hs
                        for sim_t in similar_tails:
                            for sim_e in e_to_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) and self.direct_path(sim_e,
                                                                                                           sim_rel,
                                                                                                           sim_h,
                                                                                                           hr_t) and self.direct_path(
                                    sim_t, r, sim_e, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])
                    if expl.support_paths:  # solo se si è trovato almeno un supporto si andrà a salvare la spiegazione
                        with lock:
                            paths_expl[4].append(expl)

        # type 5 and 6
        # (h --rs--> e' --r'--> t)
        for e in ent_from_h:  # e' = una qualsiasi entità nel mezzo puntata da h
            for r in rel_ingoing_t:  # per ogni relazione entrante in t
                # paths_expl[5] = paths_expl[5] + self.direct_path(tail, r, e, tr_h)
                if self.direct_path(tail, r, e, tr_h):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_from_hs = self.__tails(sim_h, sim_rel, hr_t)  # entità a cui punta hs
                        for sim_t in similar_tails:
                            for sim_e in e_from_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) and self.direct_path(sim_h,
                                                                                                           sim_rel,
                                                                                                           sim_e,
                                                                                                           hr_t) and self.direct_path(
                                    sim_e, r, sim_t, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])
                    if expl.support_paths:  # solo se si è trovato almeno un supporto si andrà a salvare la spiegazione
                        with lock:
                            paths_expl[5].append(expl)

            for r in rel_outgoing_t:  # per ogni relazione uscente da t
                # paths_expl[6] = paths_expl[6] + self.direct_path(tail, r, e, hr_t)
                if self.direct_path(tail, r, e, hr_t):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_from_hs = self.__tails(sim_h, sim_rel, hr_t)  # entità a cui punta hs
                        for sim_t in similar_tails:
                            for sim_e in e_from_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) and self.direct_path(sim_h,
                                                                                                           sim_rel,
                                                                                                           sim_e,
                                                                                                           hr_t) and self.direct_path(
                                    sim_t, r, sim_e, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])
                    if expl.support_paths:  # solo se si è trovato almeno un supporto si andrà a salvare la spiegazione
                        with lock:
                            paths_expl[6].append(expl)


def pretty_print(paths_dict, data: DataManager):
    """
    Print the predictions and the explanations for each of them, if exist
    :param paths_dict: dictionary containing the explanations evaluated; it has to be in the form {test_triple1: {pred_id : [expl_obj, expl_obj,...], ...}
    :param data:
    :return:
    """
    log = Log.get_logger()
    for triple_test_index in paths_dict.keys():
        test_triple = data.test_triples[triple_test_index]
        log.debug(f"Tripla di test: {test_triple}")
        head = test_triple[0]
        rel = test_triple[2]
        for pred_index in paths_dict[triple_test_index].keys():
            log.debug(
                f"\tSpiegazioni per la predizione ({head} --{rel}--> {data.test_predicted_tails[triple_test_index][pred_index]})")
            explanations = paths_dict[triple_test_index][pred_index]
            if explanations != {None}:
                # siamo nel dizionario che contiene le diverse tipologie di path
                log.debug(explanations)
                for k in explanations.keys():
                    if explanations[k]:  # = se non è vuota la lista di quella tipologia di spiegazioni
                        log.debug(f"Type {k}:")
                        [log.debug(e) for e in explanations[k]]



def evaluation(paths_dict=None):
    """
    Evaluates recall and average support
    :return:
    """
    predictions = 0
    predictions_with_expl = 0  # number of predictions with at least 1 explanation supported
    supp_per_type = {}
    expl_per_type = {}
    for triple_test_index in paths_dict.keys():
        """test_triple = data.test_triples[triple_test_index]
        print(f"Tripla di test: {test_triple}")
        head = test_triple[0]
        rel = test_triple[2]"""

        for pred_index in paths_dict[triple_test_index].keys():
            # RECALL
            # predizioni totali/predizioni con almeno un path di spiegazione (il quale è vincolato ad avere almeno un supporto)
            predictions += 1  # serve il totale delle predizioni
            explanations = paths_dict[triple_test_index][pred_index]
            if explanations != {None}:
                predictions_with_expl += 1
                # siamo nel dizionario che contiene le diverse tipologie di path
                #print(explanations)
                for type in explanations.keys():
                    if explanations[type]:  # = se non è vuota la lista di quella tipologia di spiegazioni
                        #print(f"Type {type}:")
                        #[print(e) for e in explanations[type]]
                        num_expl = 0 # number of explanations of this type for the given prediction
                        for e in explanations[type]: # per ogni spiegazione prendo il supporto ottenuto, che sarà almeno 1
                            num_expl +=1
                            try:
                                supp_per_type[type] += len(e.support_paths) # qui conta quanto supporto per questo tipo per questa spiegazione, che si somma al totale supporti di questa tipologia
                            except KeyError:
                                # dizionario alla prima iterazione
                                supp_per_type[type] = len(e.support_paths)
                        try:
                            expl_per_type[type] += num_expl # tutte le spiegazioni di questo tipo
                        except KeyError:
                            expl_per_type[type] = num_expl


    recall = predictions_with_expl/predictions
    #print(recall)

    # avg support
    # supporto medio sulle triple per cui vi è almeno una spiegazione
    avg_supp_per_type = {}
    for type in expl_per_type.keys():
        avg_supp_per_type[type] = supp_per_type[type]/expl_per_type[type]
    return recall, avg_supp_per_type


def main_process(data: DataManager, num_tripla: int, explainer: Explainer, return_dict, args):
    """
    Processo adibito alla generazione di spiegazioni per la tripla num_tripla; utilizzato in multiprocessing per parallelizzare
    la generazione di spiegazioni per più triple contemporaneamente
    :param data:
    :param num_tripla:
    :param explainer:
    :param return_dict:
    :return:
    """
    log = Log.get_logger()
    tripla_test = data.test_triples[num_tripla]
    log.info(f"Processing predictions for test triple: {tripla_test}")
    # ids
    test_head_id = tripla_test[0]
    test_tail_id = tripla_test[1]
    rel_id = tripla_test[2]
    # embeddings of the test triple
    head_emb = data.entity_emb[test_head_id]
    tail_emb = data.entity_emb[test_tail_id]
    rel_emb = data.rel_emb[rel_id]
    inv_rel_emb = data.inv_rel[rel_id]
    ## TAIL PREDICTION EXPLANATION
    tail_predictions = data.test_predicted_tails[num_tripla]
    # similarità con il rel_emb della tripla
    sim_rels = explainer.top_sim_emb(rel_id, data.relations_similarities_dict, top_k=args.top_rel)
    sim_heads = explainer.top_sim_emb(test_head_id, data.entities_similarities_dict, top_k=args.top_ent)

    paths_for_pred = {}  # dict contenente {num_pred: paths, num_pred1: path1} k = indice per tail_predictions, v = prediction_paths

    for num_pred in range(0, len(tail_predictions)):
        predicted_tail_id = tail_predictions[num_pred]
        # le code simili servono per la ricerca di spiegazioni a supporto
        sim_tails = explainer.top_sim_emb(predicted_tail_id, data.entities_similarities_dict,
                                          top_k=10)
        # dunque cercare spiegazione per (head_id, pred_tail, rel_id)
        paths_for_pred[num_pred] = explainer.paths(test_head_id, rel_id, sim_rels, predicted_tail_id,
                                                   [data.train_hr_t, data.train_tr_h], sim_heads, sim_tails)

    return_dict[num_tripla] = paths_for_pred


def main(manager):
    dataset = DataManager(args.data_dir, args.pred_perc)
    log.info('Data loaded')
    log.info("NB: triples expressed in the form [h,t,r], but explanations and paths will be in the canonical form [h,r,t]\n")
    explainer = Explainer()
    #manager = multiprocessing.Manager()
    paths_dictionary = manager.dict()
    jobs = []
    # {num_tripla: {num_predizione: {paths} } }
    max_processes = args.max_processes
    actual_processes = 0
    for num_tripla in range(0, len(dataset.test_triples)):
        p = multiprocessing.Process(target=main_process, args=(dataset, num_tripla, explainer, paths_dictionary, args))
        jobs.append(p)
        p.start()
        actual_processes += 1
        # paths_dictionary[num_tripla] = paths_for_pred
        if actual_processes == max_processes:
            for proc in jobs:
                proc.join()
            actual_processes = 0
            jobs = []

    if jobs:  # se ce ne sono ancora da concludere
        for proc in jobs:
            proc.join()

    log.info("Explanations computed.")
    log.info("Computing the performances evaluation")
    if args.pretty_print:
        print("Pretty print on the log file is running...")
        pretty_print(paths_dict=paths_dictionary, data=dataset)
        print("Pretty print is over")

    log.info("Computing the performances evaluation")
    recall, avg_sup_type = evaluation(paths_dictionary)
    log.info(f"Recall: {recall}")
    log.info(f"Avg support for each type of explantion (average support for each explanation, averaged for each type: {avg_sup_type}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossE.')
    parser.add_argument('--top_ent', dest='top_ent', type=int, help='Number of the top k similar embeddings to consider'
                                                                    'while computing the paths based on similar heads/tails',
                        default=10)
    parser.add_argument('--top_rel', dest='top_rel', type=int, help='Number of the top k similar embeddings to consider'
                                                                    'while computing the paths based on similar relationships',
                        default=5)
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder containing the output of the training phase and"
                                                                  "the similarities between entites and between relationships (pickle files)")
    parser.add_argument('--log_level', dest='log_level', type=str,
                        help='set the logging level, choose between info or debug',
                        default="info")
    parser.add_argument('--processes', dest='max_processes', type=int,
                        help='number of processes on which to parallelize the computation',
                        default=10)

    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        help='directory to save in the logs with the performances output and explanations',
                        default=r"expl_results/")

    parser.add_argument('--predictions_perc', dest='pred_perc', type=float,
                        help='percentage of the predictions to take into account for each test triple',
                        default=2)

    parser.add_argument('--distance', dest='distance_type', type=str,
                        help='choose the pre-computed distance/similarity to involve: euclidian, cosine',
                        default='euclidian')

    parser.add_argument('--pretty_print', dest='pretty_print_flag', type=bool,
                        help='if true, enable the pretty print of the explanations on the log file (requires much time',
                        default=False)

    global args
    args = parser.parse_args()
    log_save_dir = f"{args.save_dir}execution_logs"  # to save a subfolder with the fraction used
    Path(log_save_dir).mkdir(parents=True, exist_ok=True)

    if args.log_level == "debug":
        log = Log.get_logger(logs_dir=log_save_dir, level=Log.Levels.DEBUG)
    else:
        log = Log.get_logger(logs_dir=log_save_dir)

    log.info("DATA DIR: %s" % args.data_dir)
    log.info("TOP_ENT: %d" % args.top_ent)
    log.info("TOP_REL: %d" % args.top_rel)
    log.info("LOG LEVEL: %s" % args.log_level)
    log.info("PROCESSES: %d" % args.max_processes)
    log.info("SAVE DIR: %s" % args.save_dir)
    log.info("PERCENTAGE OF PREDICTIONS: %d" % args.pred_perc)
    log.info("DISTANCE TYPE: %s" % args.distance_type)
    global manager # dovrebbe aiutare con la interruzione prematura dei processi
    manager = multiprocessing.Manager() # manager for the shared dict in multiprocessing
    start_time = time.time()  # better for windows, more accuracy
    main(manager)
    log.debug("--- %s seconds ---" % (time.time() - start_time))
