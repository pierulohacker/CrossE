import argparse
import multiprocessing
import pickle
from global_logger import Log
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from threading import Thread, RLock


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))

class DataManager():
    @staticmethod
    def reduce_predictions(list_of_pred, percentage):
        """
        Reduce the list of arrays o predictions given in input maintaining only a specified percentage of predictions
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

    def __init__(self, pickles_path, percentage_predictions, clustering):
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

        file_path = pickles_path + file_names[5]
        with open(file_path, 'rb') as f:
            self.__test_predicted_tails = pickle.load(f)
        # reduction of tail predictions
        self.__test_predicted_tails = DataManager.reduce_predictions(self.__test_predicted_tails,
                                                                     percentage_predictions)

        file_path = pickles_path + file_names[7]
        with open(file_path, 'rb') as f:
            self.__test_triples = pickle.load(f)

        file_path = pickles_path + file_names[8]
        with open(file_path, 'rb') as f:
            self.__train_hr_t = pickle.load(f)

        file_path = pickles_path + file_names[9]
        with open(file_path, 'rb') as f:
            self.__train_tr_h = pickle.load(f)

        if clustering:
            similarity_data_path = f"{pickles_path}{clustering}/{args.distance_type}/"
        else:
            similarity_data_path = f"{pickles_path}{args.distance_type}/"
        file_path = similarity_data_path + file_names[10]
        with open(file_path, 'rb') as f:
            self.__entity_id_sim = pickle.load(f)

        file_path = similarity_data_path + file_names[11]
        with open(file_path, 'rb') as f:
            self.__relation_id_sim = pickle.load(f)

        if args.distance_type != "semantic":
            file_path = similarity_data_path + file_names[12]
            with open(file_path, 'rb') as f:
                self.__inv_rel_id_sim = pickle.load(f)

    """@property
    def entity_emb(self):
        return self.__entity_emb

    @property
    def inv_rel(self):
        return self.__inv_rel

    @property
    def rel_emb(self):
        return self.__rel_emb"""

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
        Nested class used for creating explanation object: an explanation object is a triple (h, r, t) or a path
        (h,r,e,r',t) with a series of triples that support the explanation
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
            """Print an explaanation obnect as explanation - similar example - support"""
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

    def paths(self, head, relationship, sim_relationships, tail, train_dicts: list, similar_heads, similar_tails, args):
        """
        Looks for paths between head and tail via relationships provided in input; there are 6 paths that can be detected
        :param similar_tails:
        :param similar_heads:
        :param head_emb:
        :param tail_emb:
        :param emb_matrix:
        :param relationship: id of the relationship of the triple
        :param sim_relationships: list of similar relationships for which to find a path
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
        """
        paths_expl will contain the explanations of 6 different typologies for each prediction of the input triple;
        for type 1 it will contain a list of 1-triples:[[h,rs,t]]. The same applies for type 2. For type 3 we will have
        [[h,rs,e',r',t], [h,rs,e',r',t],...], which are read as: h <--rs-- e' --r'--> t
        """
        lock = RLock()  # to manage resources in the multithread path finder
        actual_threads = 0
        max_threads = 8
        if args.multithreading is True:
            thread_list = []
            for sim_rel in sim_relationships:
                t = Thread(target=self.__multithread_path_finder,
                           args=(head, sim_rel, tail, hr_t, similar_heads, similar_tails, relationship,
                                 paths_expl, lock, tr_h))
                t.start()
                thread_list.append(t)
                # multithread_path_find
                actual_threads += 1
                if actual_threads == max_threads:
                    for thread in thread_list:
                        thread.join()
                    actual_threads = 0
                    thread_list = []

            if thread_list:  # se ce ne sono ancora da concludere
                for thread in thread_list:
                    thread.join()
        else:
            for sim_rel in sim_relationships:
                self.__multithread_path_finder(head, sim_rel, tail, hr_t, similar_heads, similar_tails, relationship,
                                               paths_expl, lock, tr_h)

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
                    # check for direct link between the similar entities through the original relationship
                    if self.direct_path(sim_h, relationship, sim_t, hr_t) \
                            and self.direct_path(sim_h, sim_rel, sim_t, hr_t):
                        expl.add_support_path([sim_h, relationship, sim_t], [sim_h, sim_rel, sim_t])
            if expl.support_paths:  # if at least one support was found, the explanation is saved
                with lock:
                    paths_expl[1].append(expl)

        # ("\nType 2:")
        if self.direct_path(tail, sim_rel, head, hr_t):
            expl = self.Explanation([tail, sim_rel, head])
            # FIND SUPPORT
            for sim_h in similar_heads:
                for sim_t in similar_tails:
                    # check for direct link between the similar entities through the original relationship + among them with similar rel
                    if self.direct_path(sim_h, relationship, sim_t, hr_t) \
                            and self.direct_path(sim_t, sim_rel, sim_h, hr_t):
                        expl.add_support_path([sim_h, relationship, sim_t], [sim_t, sim_rel, sim_h])

            if expl.support_paths:
                with lock:
                    paths_expl[2].append(expl)

        ent_to_h = self.__tails(head, sim_rel,
                                tr_h)  # NB __tails in this case takes all the entities that point at h
        ent_from_h = self.__tails(head, sim_rel, hr_t)  # entities pointed by h through rel
        rel_ingoing_t = self.__relations(tail, tr_h)  # incoming relationships in tail
        rel_outgoing_t = self.__relations(tail, hr_t)  # outgoing relationships from tail

        # Type 3 (h <--rs-- e' --r'--> t)
        for e in ent_to_h:
            # top 10 similar to e and connected to hs (intersezione tra le es e le uscenti da hs
            # ent_sim_e_to_hs = set(self.top_sim_emb(emb_matrix[e], e, emb_matrix, top_k)).intersection()
            for r in rel_ingoing_t:  # per ogni relazione entrante in t
                if self.direct_path(tail, r, e, tr_h):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_to_hs = self.__tails(sim_h, sim_rel, tr_h)  # entities going in hs
                        for sim_t in similar_tails:
                            for sim_e in e_to_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) \
                                        and self.direct_path(sim_e, sim_rel, sim_h, hr_t) \
                                        and self.direct_path(sim_e, r, sim_t, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])

                    if expl.support_paths:
                        with lock:
                            paths_expl[3].append(expl)
            # (h <--rs-- e' <--r'-- t) tipo 4 together with type 3 since the first for is the same
            for r in rel_outgoing_t:
                # paths_expl[4] = paths_expl[4] + self.direct_path(tail, r, e, hr_t)
                if self.direct_path(tail, r, e, hr_t):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_to_hs = self.__tails(sim_h, sim_rel, tr_h)
                        for sim_t in similar_tails:
                            for sim_e in e_to_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) \
                                        and self.direct_path(sim_e, sim_rel, sim_h, hr_t) \
                                        and self.direct_path(sim_t, r, sim_e, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])
                    if expl.support_paths:
                        with lock:
                            paths_expl[4].append(expl)

        # type 5 and 6
        # (h --rs--> e' --r'--> t)
        for e in ent_from_h:  # e' = any in-between entity pointed by h
            for r in rel_ingoing_t:  # for each incoming relationship in t
                # paths_expl[5] = paths_expl[5] + self.direct_path(tail, r, e, tr_h)
                if self.direct_path(tail, r, e, tr_h):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_from_hs = self.__tails(sim_h, sim_rel, hr_t)  # entity to which hs points
                        for sim_t in similar_tails:
                            for sim_e in e_from_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) \
                                        and self.direct_path(sim_h, sim_rel, sim_e, hr_t) \
                                        and self.direct_path(sim_e, r, sim_t, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])
                    if expl.support_paths:
                        with lock:
                            paths_expl[5].append(expl)

            for r in rel_outgoing_t:
                # paths_expl[6] = paths_expl[6] + self.direct_path(tail, r, e, hr_t)
                if self.direct_path(tail, r, e, hr_t):
                    expl = self.Explanation([head, sim_rel, e, r, tail])
                    for sim_h in similar_heads:
                        e_from_hs = self.__tails(sim_h, sim_rel, hr_t)
                        for sim_t in similar_tails:
                            for sim_e in e_from_hs:
                                if self.direct_path(sim_h, relationship, sim_t, hr_t) \
                                        and self.direct_path(sim_h, sim_rel, sim_e, hr_t) \
                                        and self.direct_path(sim_t, r, sim_e, hr_t):
                                    expl.add_support_path([sim_h, relationship, sim_t],
                                                          [sim_h, sim_rel, sim_e, r, sim_t])
                    if expl.support_paths:
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
            explanations = paths_dict[triple_test_index][pred_index]
            if explanations != {None}:
                log.debug(
                    f"\tSpiegazioni per la predizione ({head} --{rel}--> {data.test_predicted_tails[triple_test_index][pred_index]})")
                # dictionary that contains the different path typologies
                log.debug(explanations)
                for k in explanations.keys():
                    if explanations[k]:
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
            # RECALL = |predictions|/|predictions with at least one explanation path|
            predictions += 1
            explanations = paths_dict[triple_test_index][pred_index]
            if explanations != {None}:
                predictions_with_expl += 1
                for type in explanations.keys():
                    if explanations[type]:
                        num_expl = 0  # number of explanations of this type for the given prediction
                        for e in explanations[type]:
                            num_expl += 1
                            try:
                                supp_per_type[type] += len(e.support_paths)  # counts how much support of this type for this explanation, which is summed to the total amount of supports for this typology
                            except KeyError:
                                supp_per_type[type] = len(e.support_paths)
                        try:
                            expl_per_type[type] += num_expl  # all the explanations of this type
                        except KeyError:
                            expl_per_type[type] = num_expl

    recall = predictions_with_expl / predictions

    # avg support on the triples for which thes is at least one explanation
    avg_supp_per_type = {}
    for type in expl_per_type.keys():
        avg_supp_per_type[type] = supp_per_type[type] / expl_per_type[type]
    return recall, avg_supp_per_type


def main_process(data: DataManager, num_tripla: int, explainer: Explainer, return_dict, args):
    """
    Process that generates explanation for the triple num_tripla; used in multiprocessing to parallelize the generation
    of explanation for more triples at the same time
    :param data:
    :param num_tripla:
    :param explainer:
    :param return_dict:
    :return:
    """
    log = Log.get_logger()
    tripla_test = data.test_triples[num_tripla]
    test_head_id = tripla_test[0]
    rel_id = tripla_test[2]
    # embeddings of the test triple
    """head_emb = data.entity_emb[test_head_id]
    tail_emb = data.entity_emb[test_tail_id]
    rel_emb = data.rel_emb[rel_id]
    inv_rel_emb = data.inv_rel[rel_id]"""
    ## TAIL PREDICTION EXPLANATION
    tail_predictions = data.test_predicted_tails[num_tripla]
    # similarity with the triple relational embedding
    sim_rels = explainer.top_sim_emb(rel_id, data.relations_similarities_dict, top_k=args.top_rel)
    sim_heads = explainer.top_sim_emb(test_head_id, data.entities_similarities_dict, top_k=args.top_ent)

    paths_for_pred = {}  # dict having {num_pred: paths, num_pred1: path1} k = index for tail_predictions, v = prediction_paths
    for num_pred in range(0, len(tail_predictions)):
        predicted_tail_id = tail_predictions[num_pred]
        # similar tails are used for searching for support explanations
        sim_tails = explainer.top_sim_emb(predicted_tail_id, data.entities_similarities_dict,
                                          top_k=args.top_ent)
        # look for explanation for (head_id, pred_tail, rel_id)
        paths_for_pred[num_pred] = explainer.paths(test_head_id, rel_id, sim_rels, predicted_tail_id,
                                                   [data.train_hr_t, data.train_tr_h], sim_heads, sim_tails, args)

    return_dict[num_tripla] = paths_for_pred


def main(manager):
    dataset = DataManager(args.data_dir, args.pred_perc, args.clustering)
    print(len(dataset.test_triples))
    log.info('Data loaded')
    log.info(
        "NB: triples expressed in the form [h,t,r], but explanations and paths will be in the canonical form [h,r,t]\n")
    explainer = Explainer()
    if args.multiproc_flag: # If multiprocessing is enabled
        paths_dictionary = manager.dict()
        jobs = []
        max_processes = args.max_processes
        actual_processes = 0
        for num_tripla in range(0, len(dataset.test_triples)):
            p = multiprocessing.Process(target=main_process,
                                        args=(dataset, num_tripla, explainer, paths_dictionary, args))
            jobs.append(p)
            p.start()
            multiprocessing.Process  # should prevent excessive RAM waste in Linux https://stackoverflow.com/a/14750086/9748304
            actual_processes += 1
            if actual_processes == max_processes:
                for proc in jobs:
                    proc.join()
                actual_processes = 0
                jobs = []

        if jobs:
            for proc in jobs:
                proc.join()

    else:
        paths_dictionary = dict()
        for num_tripla in tqdm(range(0, len(dataset.test_triples))):
            main_process(dataset, num_tripla, explainer, paths_dictionary, args)

    log.info("Explanations computed.")
    log.info("Computing the performances evaluation")

    recall, avg_sup_type = evaluation(paths_dictionary)
    with open("results.txt", "w") as f:
        f.write(f"Recall: {recall}")
        f.write(f"Avg support for each type of explantion (average support for each explanation, averaged for each type: {avg_sup_type}")

    log.info(f"Recall: {recall}")
    log.info(
        f"Avg support for each type of explantion (average support for each explanation, averaged for each type: {avg_sup_type}")
    if args.pretty_print_flag:
        print("Pretty print on the log file is running...")
        pretty_print(paths_dict=paths_dictionary, data=dataset)
        print("Pretty print is over")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossE.')
    parser.add_argument('--top_ent', dest='top_ent', type=int, help='Number of the top k similar embeddings to consider'
                                                                    'while computing the paths based on similar heads/tails',
                        default=10)
    parser.add_argument('--top_rel', dest='top_rel', type=int, help='Number of the top k similar embeddings to consider'
                                                                    'while computing the paths based on similar relationships',
                        default=5)
    parser.add_argument('--data', dest='data_dir', type=str,
                        help="Data folder containing the output of the training phase and"
                             "the similarities between entites and between relationships (pickle files)")
    parser.add_argument('--log_level', dest='log_level', type=str,
                        help='set the logging level, choose between info or debug',
                        default="info")
    parser.add_argument('--multiprocessing', dest='multiproc_flag', type=bool,
                        help='enables multiprocessing, by default is not enabled',
                        default=False)
    parser.add_argument('--multithreading', dest='multithreading', type=bool,
                        help='enables multithreading to compute paths for explanation, by default is not enabled',
                        default=False)
    parser.add_argument('--processes', dest='max_processes', type=int,
                        help='number of processes on which to parallelize the computation',
                        default=2)

    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        help='directory to save in the logs with the performances output and explanations',
                        default=r"expl_results/")

    parser.add_argument('--predictions_perc', dest='pred_perc', type=float,
                        help='percentage of the predictions to take into account for each test triple',
                        default=2)

    parser.add_argument('--distance', dest='distance_type', type=str,
                        help='choose the pre-computed distance/similarity to involve: euclidian, cosine',
                        default='euclidian')

    parser.add_argument('--clustering', dest='clustering', type=str,
                        help='Name of the folder containing the similarities files obtained through clustering. If you'
                             'don\t want to use clustering, leave this argument as none',
                        default=None)

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
    log.info("MULTIPROCESSING: %s" % args.multiproc_flag)
    if args.multiproc_flag is True:
        log.info("PROCESSES: %d" % args.max_processes)
    log.info("SAVE DIR: %s" % args.save_dir)
    log.info("PERCENTAGE OF PREDICTIONS: %s" % args.pred_perc)
    log.info("DISTANCE TYPE: %s" % args.distance_type)
    global manager  # should help with premature termination of processes
    manager = multiprocessing.Manager()  # manager for the shared dict in multiprocessing
    main(manager)

# python3 explanation/explanation.py --data ./save/FB15K/out_data/pickle/ --clustering agglomerative/10 --save_dir explanation/results/WN18/euclidian/8kmeans/2perc/ --distance euclidian --predictions_perc 2
