import argparse
import math
import os
import os.path
import pickle
import timeit
from multiprocessing import JoinableQueue, Queue, Process
from pathlib import Path

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from global_logger import Log
from varname import nameof

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CrossE:
    """trainable variables of the model (it's just a declaration for documentation purposes)"""
    # Matrices
    __ent_embedding: tf.Variable  # E
    __rel_embedding: tf.Variable  # R
    __rel_embedding_reverse: tf.Variable  # R for reverse relations (to train the model on triples (t, r^-1, h) )

    # combination matrix refferred to tail entities interaction
    __t_weighted_vector: tf.Variable
    # combination matrix refferred to head entities interaction
    __h_weighted_vector: tf.Variable
    # combination biases
    __tr_combination_bias: tf.Variable  # tf.variable name will be "simple_t_combination_weights"
    __hr_combination_bias: tf.Variable  # tf.variable name will be "simple_h_combination_weights"
    """#####################################################################################"""

    @property
    def n_entity(self):
        return self.__n_entity

    @property
    def n_train(self):
        return self.__train_triple.shape[0]

    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def hr_t(self):
        return self.__hr_t

    @property
    def tr_h(self):
        return self.__tr_h

    @property
    def train_tr_h(self):
        return self.__train_tr_h

    @property
    def train_hr_t(self):
        return self.__train_hr_t
    @property
    def test_triples(self):
        """
        Return the whole test set processed as triples in list; different from yeld_testing_data
        :return:
        """
        return self.__test_triple

    @property
    def test_hr_t(self):
        return self.__test_hr_t

    @property
    def test_tr_h(self):
        return self.__test_tr_h

    @staticmethod
    def out_ent_emb():
        return CrossE.__ent_embedding

    @property
    def ent_embedding(self):
        """
        Used to contain the embeddings of the entities
        :return: tensor of entity embeddings
        """
        return self.__ent_embedding



    @property
    def rel_embedding(self):
        """
        Getter for the relational embeddings
        :return: relational embeddings tensor variable
        """
        return self.__rel_embedding

    @property
    def reverse_rel_embedding(self):
        """
        :return: tensor containing inverse relationships embeddings
        """
        return self.__rel_embedding_reverse

    def yeld_raw_training_data(self, batch_size=100):
        """
        Used in iterations (yeld, so with generators) returns train triple
        :param batch_size:
        :return:
        """
        n_triple = len(self.__train_triple)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start = end


    def yeld_testing_data(self, batch_size=100):
        """
        Gives to the generator a batch of the testing triples
        :param batch_size: batch size
        :return:
        """
        n_triple = len(self.__test_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__test_triple[start:end, :]
            start = end

    def valid_data(self, batch_size=100):
        """
        used with a generator, returns each time 'batch_size' evaluation data
        :param batch_size: number of data to return to the generator
        :return:
        """
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end

    def yield_ent_embeddings(self, batch_size=100):
        """
        used with a generator, returns each time 'batch_size' entity embeddings
        :param batch_size: number of data to yeld and return to the generator
        :return:
        """
        n_emb = self.__ent_embedding.shape[0]
        start = 0
        while start < n_emb:
            end = min(start + batch_size, n_emb)
            yield self.__ent_embedding[start:end, :]
            start = end

    def yield_rel_embeddings(self, batch_size=100):
        """
        used with a generator, returns each time 'batch_size' relation embeddings
        :param batch_size: number of data to yeld and return to the generator
        :return:
        """
        n_emb = self.__rel_embedding.shape[0]
        start = 0
        while start < n_emb:
            end = min(start + batch_size, n_emb)
            yield self.__rel_embedding[start:end, :]
            start = end

    def yield_reverse_rel_embeddings(self, batch_size=100):
        """
        used with a generator, returns each time 'batch_size' reverse relation embeddings
        :param batch_size: number of data to yeld and return to the generator
        :return:
        """
        n_emb = self.__rel_embedding_reverse.shape[0]
        start = 0
        while start < n_emb:
            end = min(start + batch_size, n_emb)
            yield self.__rel_embedding_reverse[start:end, :]
            start = end

    def __init__(self, data_dir=None, embed_dim=100, combination_method='simple', dropout=0.5, neg_weight=0.5):
        """
        Create an instance of a model (CrossE)
        :param data_dir: directory containing the data; args
        :param embed_dim: dimension of the embeddings; args
        :param combination_method: method that combines the embeddigns; args
        :param dropout: value of the dropout, between 0 and 1.0; args
        :param neg_weight:
        """
        # create and/or reuse a global logger, choosing its name dynamicaly
        # with screen-only output and the default logging level INFO
        log = Log.get_logger()
        self.__combination_method = combination_method

        self.__embed_dim = embed_dim
        self.__initialized = False

        self.__trainable = list()  # list that will contain the trainable parameters of the model
        self.__dropout = dropout
        # loading entities and id mappings
        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.__n_entity = len(f.readlines())

        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.__entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}

        log.info("N_ENTITY: %d" % self.__n_entity)

        # loading relationships and id mappings
        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__n_relation = len(f.readlines())

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__relation_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_relation_map = {v: k for k, v in self.__entity_id_map.items()}

        log.info("N_RELATION: %d" % self.__n_relation)

        def load_triple(file_path):
            """
            Nested function; Load the triples containined in the input file path
            :param file_path: path containing the triples to load; used for train, validation, and test sets.
            The triples have to be in the form "head    tail    relation"
            :return:
            """
            with open(file_path, 'r', encoding='utf-8') as f_triple:
                return np.asarray([[self.__entity_id_map[x.strip().split('\t')[0]],
                                    self.__entity_id_map[x.strip().split('\t')[1]],
                                    self.__relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
                                  dtype=np.int32)

        def gen_hr_t(triple_data):
            """
            Nested function; Generates dictionary containing triples in the order (h,r,t)
            :param triple_data: triples to process
            :return: nested dictionary containing {head: {rel : {tail1,tail2}, rel2: {tailX, tailY} }}
            """
            hr_t = dict()
            for h, t, r in triple_data:
                if h not in hr_t:
                    hr_t[h] = dict()
                if r not in hr_t[h]:
                    hr_t[h][r] = set()  # dunque un dizionario in ogni cella, contenente un set di entità coda
                hr_t[h][r].add(t)

            return hr_t

        def gen_tr_h(triple_data):
            """
            Generates dictionary containing triples in the order (t,r,h)
            :param triple_data: triples to process
            :return: nested dictionary containing {tail: {rel : {head1,head2}, rel2: {headX, headY} }}
            """
            tr_h = dict()
            for h, t, r in triple_data:
                if t not in tr_h:
                    tr_h[t] = dict()
                if r not in tr_h[t]:
                    tr_h[t][r] = set()
                tr_h[t][r].add(h)
            return tr_h

        # loading of the dataset splits
        self.__train_triple = load_triple(os.path.join(data_dir, 'train.txt'))
        log.info("N_TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
        log.info("N_TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))
        log.info("N_VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        self.__train_hr_t = gen_hr_t(self.__train_triple)
        self.__train_tr_h = gen_tr_h(self.__train_triple)
        self.__test_hr_t = gen_hr_t(self.__test_triple)
        self.__test_tr_h = gen_tr_h(self.__test_triple)

        self.__hr_t = gen_hr_t(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))
        self.__tr_h = gen_tr_h(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))

        bound = 6 / math.sqrt(embed_dim)

        with tf.device('/cpu'):
            # initializing the embeddings of the entities
            self.__ent_embedding = tf.get_variable("ent_embedding", [self.__n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=345))
            self.__trainable.append(self.__ent_embedding)
            # initializing the embeddings of the relations
            self.__rel_embedding = tf.get_variable("rel_embedding", [self.__n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=346))
            # initializing the embeddings of the inverted relations, to model triples also in the form (t,r^-1, h)
            self.__rel_embedding_reverse = tf.get_variable("rel_embedding_reverse", [self.__n_relation, embed_dim],
                                                           initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                     maxval=bound,
                                                                                                     seed=347))
            self.__trainable.append(self.__rel_embedding)
            self.__trainable.append(self.__rel_embedding_reverse)

            # combination matrix refferred to head entities interaction
            self.__h_weighted_vector = tf.get_variable("simple_h_combination_weights",
                                                       shape=[self.__n_relation, embed_dim],
                                                       initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                 maxval=bound,
                                                                                                 seed=445))
            # combination matrix refferred to tail entities interaction
            self.__t_weighted_vector = tf.get_variable("simple_t_combination_weights",
                                                       shape=[self.__n_relation, embed_dim],
                                                       initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                 maxval=bound,
                                                                                                 seed=445))

            self.__trainable.append(self.__h_weighted_vector)
            self.__trainable.append(self.__t_weighted_vector)

            self.__hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                         initializer=tf.zeros([embed_dim]))
            self.__tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                         initializer=tf.zeros([embed_dim]))

            self.__trainable.append(self.__hr_combination_bias)
            self.__trainable.append(self.__tr_combination_bias)

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor * tf.abs(weights), 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp * tf.abs(weights), 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.

    def train(self, inputs, regularizer_weight=1., scope=None):
        """
        Method called from train_ops; trains the model starting from the given inputs
        :param inputs: list of tensors
        :param regularizer_weight:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()
            rel_embedding = self.__rel_embedding
            rel_embedding_reverse = self.__rel_embedding_reverse
            normalized_ent_embedding = self.__ent_embedding
            h_weighted_vector = self.__h_weighted_vector
            t_weighted_vector = self.__t_weighted_vector

            hr_tlist, hr_tlist_weight, tr_hlist, tr_hlist_weight = inputs

            # (?, dim)
            hr_tlist_h = tf.nn.embedding_lookup(normalized_ent_embedding, hr_tlist[:, 0])
            hr_tlist_r = tf.nn.embedding_lookup(rel_embedding, hr_tlist[:, 1])
            # (?, dim)
            tr_hlist_t = tf.nn.embedding_lookup(normalized_ent_embedding, tr_hlist[:, 0])
            tr_hlist_r = tf.nn.embedding_lookup(rel_embedding_reverse, tr_hlist[:, 1])

            h_weight = tf.nn.embedding_lookup(h_weighted_vector, hr_tlist[:, 1])
            t_weight = tf.nn.embedding_lookup(t_weighted_vector, tr_hlist[:, 1])

            # shape (?, dim)
            hr_tlist_hr = hr_tlist_h * h_weight + hr_tlist_r * hr_tlist_h * h_weight
            # operatore di score f(h,r,t) prima del sigmoide
            hrt_res = tf.matmul(tf.nn.dropout(tf.tanh(hr_tlist_hr + self.__hr_combination_bias), self.__dropout),
                                self.__ent_embedding,
                                transpose_b=True)

            tr_hlist_tr = tr_hlist_t * t_weight + tr_hlist_r * tr_hlist_t * t_weight
            # operatore di score f(t,r^-1,h) prima del sigmoide
            trh_res = tf.matmul(tf.nn.dropout(tf.tanh(tr_hlist_tr + self.__tr_combination_bias), self.__dropout),
                                self.__ent_embedding,
                                transpose_b=True)

            regularizer_loss = 0.0
            for param in self.__trainable:
                regularizer_loss += tf.reduce_sum(tf.abs(param))
            self.regularizer_loss = regularizer_loss

            hrt_res_sigmoid = tf.sigmoid(hrt_res)  # the sigma operator of the paper, applied to tanh...

            hrt_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(hrt_res_sigmoid, 1e-10, 1.0)) * tf.maximum(0., hr_tlist_weight)
                + tf.log(tf.clip_by_value(1 - hrt_res_sigmoid, 1e-10, 1.0)) * tf.maximum(0.,
                                                                                         tf.negative(hr_tlist_weight)))

            trh_res_sigmoid = tf.sigmoid(trh_res)

            trh_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(trh_res_sigmoid, 1e-10, 1.0)) * tf.maximum(0., tr_hlist_weight)
                + tf.log(tf.clip_by_value(1 - trh_res_sigmoid, 1e-10, 1.0)) * tf.maximum(0.,
                                                                                         tf.negative(tr_hlist_weight)))

            return hrt_loss + trh_loss + regularizer_loss * regularizer_weight

    def test(self, inputs, scope=None):
        """
        Run testing process for the provided inputs (useful for validation; test_evaluation function compute the
        final testing phase), using the matrices created in the model which contains embeddings of entities and
        relationships; compute the scores for normal triples (h, r, t) and inverted triples (t, r^-1,
        h); this function is called by test_ops
        :param inputs:
        :param scope:
        :return: head_ids and tail_ids
        """
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()
            # lookup takes the embeddings corresponding to each test id passed trough 'imput', that is a list of triple ids
            # e.g input = [[111 120 0], [12 9 0], ... ]
            h = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 0])  # head
            t = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 1])  # tail
            r = tf.nn.embedding_lookup(self.__rel_embedding, inputs[:, 2])  # relation
            r_reverse = tf.nn.embedding_lookup(self.__rel_embedding_reverse, inputs[:, 2])  # reverse relation
            h_w = tf.nn.embedding_lookup(self.__h_weighted_vector, inputs[:, 2])
            t_w = tf.nn.embedding_lookup(self.__t_weighted_vector, inputs[:, 2])

            ent_mat = tf.transpose(self.__ent_embedding)  # entity matrix transposed

            # predict tails
            hr = h * h_w + r * h * h_w

            hrt_res = tf.sigmoid(
                tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat))  # results of the scoring function
            _, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)  # ids of the top k tails for the tail prediction

            # predict heads
            tr = t * t_w + r_reverse * t * t_w

            trh_res = tf.sigmoid(tf.matmul(tf.tanh(tr + self.__tr_combination_bias), ent_mat))
            _, head_ids = tf.nn.top_k(trh_res, k=self.__n_entity)  # ids of the top k heads for the head prediction

            return head_ids, tail_ids


def train_ops(model: CrossE, learning_rate=0.1, optimizer_str='gradient', regularizer_weight=1.0):
    """
    Train the model and optimize it
    :param model: model to optimize
    :param learning_rate: the learning rate
    :param optimizer_str: the optimization method
    :param regularizer_weight: regularizer
    :return:
    """
    with tf.device('/gpu'):
        train_hrt_input = tf.placeholder(tf.int32, [None, 2])
        train_hrt_weight = tf.placeholder(tf.float32, [None, model.n_entity])
        train_trh_input = tf.placeholder(tf.int32, [None, 2])
        train_trh_weight = tf.placeholder(tf.float32, [None, model.n_entity])

        loss = model.train([train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight],
                           regularizer_weight=regularizer_weight)
        if optimizer_str == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError("Does not support %s optimizer" % optimizer_str)

        grads = optimizer.compute_gradients(loss, model.trainable_variables)

        op_train = optimizer.apply_gradients(grads)

        return train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight, loss, op_train


def test_ops(model: CrossE):
    with tf.device('/gpu'):
        test_input = tf.placeholder(tf.int32, [None, 3])
        head_ids, tail_ids = model.test(test_input)

    return test_input, head_ids, tail_ids


def worker_func(in_queue: JoinableQueue, out_queue: Queue, hr_t, tr_h):
    """
    Uses the data contained in the in_queue to run test evaluation and put the results into the out_queue
    :param in_queue: input queue, empty at the beginning, but it will contain (testing_data, head_pred, tail_pred during
    testing phase), or None
    :param out_queue: output queue that will contain the results of the testing phase: (mean_rank_h, filtered_mean_rank_h),
    (mean_rank_t, filtered_mean_rank_t), (mrr_h, mrr_t), (fmrr_h, fmrr_t)
    :param hr_t: nested dictionary obtained with the function gen_hr_t contained in the __init__ of class CrossE
    :param tr_h: nested dictionary obtained with the function gen_tr_h contained in the __init__ of class CrossE
    :return: the out_queue is not explicitly returned, but passed via reference
    """
    while True:
        dat = in_queue.get()  # restituisce e rimuove
        if dat is None:  # when is empty or the main assign None to the queue
            in_queue.task_done()
            continue
        testing_data, head_pred, tail_pred = dat
        out_queue.put(test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h))
        in_queue.task_done()


def data_generator_func(in_queue: JoinableQueue, out_queue: Queue, train_tr_h, train_hr_t, n_entity, neg_weight):
    """
    Function, used in multiprocessing
    :param in_queue: JoinableQueue in input
    :param out_queue:
    :param train_tr_h:
    :param train_hr_t:
    :param n_entity: number of entities
    :param neg_weight:
    :return:
    """
    while True:
        dat = in_queue.get()
        if dat is None:
            break
        # [head(tail), relation, #of_total_positive_candidates, positive_instances..., negative_instances...]
        hr_tlist = list()
        hr_tweight = list()
        tr_hlist = list()
        tr_hweight = list()

        htr = dat

        for idx in range(htr.shape[0]):
            rel = htr[idx, 2]
            rel_property = 1.0

            tr_weight_tmp = np.asarray(np.random.random(n_entity) > neg_weight * rel_property, dtype='f')
            tr_weight_tmp -= 1.0
            for id in train_tr_h[htr[idx, 1]][htr[idx, 2]]:
                tr_weight_tmp[id] = 1.0
            tr_hweight.append(tr_weight_tmp)
            tr_hlist.append([htr[idx, 1], htr[idx, 2]])

            hr_weight_tmp = np.asarray(np.random.random(n_entity) > neg_weight * rel_property, dtype='f')
            hr_weight_tmp -= 1.0
            for id in train_hr_t[htr[idx, 0]][htr[idx, 2]]:
                hr_weight_tmp[id] = 1.0
            hr_tweight.append(hr_weight_tmp)
            hr_tlist.append([htr[idx, 0], htr[idx, 2]])

        out_queue.put((np.asarray(hr_tlist, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32),
                       np.asarray(tr_hlist, dtype=np.int32), np.asarray(tr_hweight, dtype=np.float32)))


def test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h):
    """
    Perform the evaluation of the performances of the model over the test data; saves, in the last iteration, the predictions
    for the test triples
    :param testing_data: a portion of data (everything works with generators) on which run the evaluation
    :param head_pred: predictions for the queries (t, r^-1, ?)
    :param tail_pred: predictions for the queries (h, r, t)
    :param hr_t: nested dictionary obtained with the function gen_hr_t contained in the __init__ of class CrossE
    :param tr_h: nested dictionary obtained with the function gen_tr_h contained in the __init__ of class CrossE
    :return:
    """
    assert len(testing_data) == len(head_pred)
    assert len(testing_data) == len(tail_pred)

    mean_rank_h = list()
    mean_rank_t = list()
    mrr_h = list()
    mrr_t = list()
    filtered_mean_rank_h = list()
    filtered_mean_rank_t = list()
    fmrr_h = list()
    fmrr_t = list()

    for i in range(len(testing_data)):
        h = testing_data[i, 0]
        t = testing_data[i, 1]
        r = testing_data[i, 2]

        # mean rank
        mr = 0
        for val in head_pred[i]:  # head_pred[i] is a list of head entities predicted, val is a head entity each cycle
            if val == h:  # if we got the right head predicted
                mean_rank_h.append(mr)
                mrr_h.append(1 / (mr + 1.0))
                break
            mr += 1

        mr = 0
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
                mrr_t.append(1 / (mr + 1.0))
            mr += 1

        # filtered mean rank
        fmr = 0
        for val in head_pred[i]:
            if val == h:
                filtered_mean_rank_h.append(fmr)
                fmrr_h.append(1 / (fmr + 1.0))
                break
            if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
                continue
            else:
                fmr += 1

        fmr = 0
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                fmrr_t.append(1 / (fmr + 1.0))
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1

    return (mean_rank_h, filtered_mean_rank_h), (mean_rank_t, filtered_mean_rank_t), (mrr_h, mrr_t), (fmrr_h, fmrr_t)


def compute_results(model: CrossE, session: tf.Session, batch: int, test_head, test_tail, test_input,
                    evaluation_queue: JoinableQueue,
                    n_worker: int, result_queue: Queue, n_iter: int, max_iter: int, output_rank: bool):
    """
    evaluate the overall performances of the model
    :param output_rank: if True, a dump of the evaluation will be created (without mean ?)
    :param max_iter: max number of iterations
    :param model: model CrossE
    :param session: session tensorflow for the current working session
    :param batch: batch size
    :param test_head:
    :param test_tail:
    :param test_input:
    :param evaluation_queue:
    :param n_worker:
    :param result_queue:
    :param n_iter: current iteration
    :return:
    """
    out_save_path_pred_head = args.save_dir + "out_data/pickle/test_predicted_heads.pkl"  # p = pickle
    out_save_path_pred_tails = args.save_dir + "out_data/pickle/test_predicted_tails.pkl"  # p = pickle
    log = Log.get_logger()  # returns the global logger
    # model.valid_data passaggio di funzione
    for data_func, test_type in zip([model.valid_data, model.yeld_testing_data], ['valid', 'TEST']):
        accu_mean_rank_h = list()
        accu_mean_rank_t = list()
        accu_mrr_h = list()
        accu_mrr_t = list()
        accu_filtered_mean_rank_h = list()
        accu_filtered_mean_rank_t = list()
        accu_fmrr_h = list()
        accu_fmrr_t = list()

        evaluation_count = 0

        save_pred_flag = False # flag to authorize the save
        # reset the files of previous executions
        if test_type == 'TEST' and n_iter == max_iter - 1:
            save_pred_flag = True

            head_predictions = []
            tails_predictions = []
        # prende un batch di validation o test,  a seconda della funzione in data_func; prende un batch di embeddings

        for testing_data in data_func(batch_size=batch):
            """
            esegue la sessione avviando la funzione richiesta da test_head e test_tail; entrambe, andando molto a ritroso,
            sono in test_ops->model.test e sono head_ids e tail_ids  che contengono gli id delle predizioni per i task di
            head prediction e tail prediction
            test_input è la chiave del 'feed_dic', esso permette di aggiornare i valori di test_input prendendo il batch
            contenuto in testing_data (id_h,id_t,id_3)
            """

            head_pred, tail_pred = session.run([test_head, test_tail],
                                               {test_input: testing_data})


            #print(f"Head predictions {head_pred.shape}")
            #print(f"Tail predictions {tail_pred.shape}")

            if save_pred_flag:
                """file = open(out_save_path_pred_head, "rb")
                old_head_pred = pickle.load(file)
                file.close()

                file = open(out_save_path_pred_tails, "rb")
                old_tail_pred = pickle.load(file)
                file.close()
                # append the head predictions
                with open(out_save_path_pred_head, "wb") as f:
                    pickle.dump(old_head_pred + head_pred, f)
                    #pickle.dump(head_pred, f)
                # append the tail predictions
                with open(out_save_path_pred_tails, "wb") as f:
                    #pickle.dump(tail_pred, f)
                    pickle.dump(old_tail_pred + tail_pred, f)"""
                # print("Concatenazione")
                # print(type(head_pred))
                for h_p in head_pred:
                    head_predictions.append(h_p)
                for t_p in tail_pred:
                    tails_predictions.append(t_p)
                """log.info(f"Predictions for the testing triples serialized to\n "
                         f"- '{out_save_path_pred_head}' for head predictions\n"
                         f"- '{out_save_path_pred_tails}' for tail predictions")"""
            evaluation_queue.put((testing_data, head_pred, tail_pred))
            # evaluation_queue.put((testing_data, head_pred, tail_pred))
            evaluation_count += 1

        # mette tanti None quanti sono i workers per arrestarli successivamente
        for i in range(n_worker):
            evaluation_queue.put(None)

        log.debug("waiting for worker finishes their work")
        evaluation_queue.join()
        log.debug("all worker stopped.")
        while evaluation_count > 0:
            evaluation_count -= 1
            (mrh, fmrh), (mrt, fmrt), (mrrh, mrrt), (fmrrh, fmrrt) = result_queue.get()
            accu_mean_rank_h += mrh
            accu_mean_rank_t += mrt
            accu_mrr_h += mrrh
            accu_mrr_t += mrrt
            accu_filtered_mean_rank_h += fmrh
            accu_filtered_mean_rank_t += fmrt
            accu_fmrr_h += fmrrh
            accu_fmrr_t += fmrrt

        if n_iter == max_iter - 1 and output_rank == True:
            pickle.dump(accu_mean_rank_h, open('./MR_h', 'wb'))
            pickle.dump(accu_mean_rank_t, open('./MR_t', 'wb'))
            pickle.dump(accu_filtered_mean_rank_h, open('./FMR_h', 'wb'))
            pickle.dump(accu_filtered_mean_rank_t, open('./FMR_t', 'wb'))

        log.info(
            "[%s] ITER %d [HEAD PREDICTION] MRR: %.3f FMRR: %.3f MR: %.1f FMR %.1f H10 %.3f FH10 %.3f H3 %.3f FH3 %.3f H1 %.3f FH1 %.3f" %
            (test_type, n_iter, np.mean(accu_mrr_h), np.mean(accu_fmrr_h),
             np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),
             np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),
             np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10),
             np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 3),
             np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 3),
             np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 1),
             np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 1)))

        log.info(
            "[%s] ITER %d [TAIL PREDICTION] MRR: %.3f FMRR: %.3f MR: %.1f FMR %.1f H10 %.3f FH10 %.3f H3 %.3f FH3 %.3f H1 %.3f FH1 %.3f" %
            (test_type, n_iter, np.mean(accu_mrr_t), np.mean(accu_fmrr_t),
             np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
             np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
             np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10),
             np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 3),
             np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 3),
             np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 1),
             np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 1)))

        log.info(
            "[%s] ITER %d [BOTH PREDICTION] MRR: %.3f FMRR: %.3f MR: %.1f FMR %.1f H10 %.3f FH10 %.3f H3 %.3f FH3 %.3f H1 %.3f FH1 %.3f" %
            (test_type, n_iter, np.mean([accu_mrr_h, accu_mrr_t]), np.mean([accu_fmrr_h, accu_fmrr_t]),
             np.mean([accu_mean_rank_t, accu_mean_rank_h]),
             np.mean([accu_filtered_mean_rank_t, accu_filtered_mean_rank_h]),
             np.mean([np.asarray(accu_mean_rank_t, dtype=np.int32) < 10,
                      np.asarray(accu_mean_rank_h, dtype=np.int32) < 10]),
             np.mean([np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10,
                      np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10]),
             np.mean([np.asarray(accu_mean_rank_t, dtype=np.int32) < 3,
                      np.asarray(accu_mean_rank_h, dtype=np.int32) < 3]),
             np.mean([np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 3,
                      np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 3]),
             np.mean([np.asarray(accu_mean_rank_t, dtype=np.int32) < 1,
                      np.asarray(accu_mean_rank_h, dtype=np.int32) < 1]),
             np.mean([np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 1,
                      np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 1])))

    # print(f"Head predictions = {len(head_predictions)}")
    # print(f"Tail predictions = {len(tails_predictions)}")
    try:
        with open(out_save_path_pred_head, "wb+") as f:
            pickle.dump(head_predictions, f)
        with open(out_save_path_pred_tails, "wb+") as f:
            pickle.dump(tails_predictions, f)
        print("predictions saved")
    except UnboundLocalError: # when the iteration is not the last to save the data
        print("skip saving of prediction")




def main(args):
    # LOG
    log_save_dir = f"{args.save_dir}execution_logs"  # to save a subfolder with the fraction used
    Path(log_save_dir).mkdir(parents=True, exist_ok=True)
    if args.log_level == "debug":
        log = Log.get_logger(logs_dir=log_save_dir, level=Log.Levels.DEBUG)
    else:
        log = Log.get_logger(logs_dir=log_save_dir)
    # paths for the serializations of the output, results, and predictions
    out_save_path_h = args.save_dir + "out_data/"  # h = human-readable
    out_save_path_p = args.save_dir + "out_data/pickle/"  # p = pickle
    Path(out_save_path_h).mkdir(parents=True, exist_ok=True)
    Path(out_save_path_p).mkdir(parents=True, exist_ok=True)

    ######
    log.info(args)
    log.info("DATASET: %s" % args.data_dir)
    log.info("DIMENSION: %d" % args.dim)
    log.info("BATCH_SIZE: %d" % args.batch)
    log.info("LEARNING_RATE: %s" % args.lr)
    log.info("NEG_WEIGHT: %s" % args.neg_weight)
    log.info("OPTIMIZER: %s" % args.optimizer)
    log.info("MAX_ITER: %d" % args.max_iter)
    log.info("EVALUATE_PER: %d" % args.eval_per)

    model = CrossE(args.data_dir, embed_dim=args.dim, combination_method=args.combination_method,
                   dropout=args.drop_out, neg_weight=args.neg_weight)

    train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight, \
    train_loss, train_op = train_ops(model, learning_rate=args.lr,
                                     optimizer_str=args.optimizer,
                                     regularizer_weight=args.loss_weight)
    test_input, test_head, test_tail = test_ops(model)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as session:
        tf.global_variables_initializer().run()
        # manage the saving of the model checkpoints and data
        saver = tf.train.Saver()

        iter_offset = 0
        # loading a model using the checkpoint in the path args.load_model, in order to restore the session and continue training
        if args.load_model is not None:
            saver.restore(session, args.load_model)
            # extracts, from the checkpoint filename, the iteration number, sum it 1 and use it to resume training
            iter_offset = int(args.load_model.split('.')[-2].split('_')[-1]) + 1
            log.info("Load model from %s, iteration %d restored." % (args.load_model, iter_offset))

        total_inst = model.n_train

        # training data generator
        raw_training_data_queue = JoinableQueue()
        training_data_queue = Queue()
        data_generators = list()
        for i in range(args.n_generator):
            data_generators.append(Process(target=data_generator_func, args=(
                raw_training_data_queue, training_data_queue, model.train_tr_h, model.train_hr_t, model.n_entity,
                float(args.neg_weight) / model.n_entity)))
            data_generators[-1].start()  # starts the last process appendend in each iteration

        evaluation_queue = JoinableQueue()
        result_queue = Queue()  # output queue of the results of the testing phase
        workers = list()
        # start of the workers that perform evaluations of the model when the evaluation_queue is populated
        for i in range(args.n_worker):
            workers.append(Process(target=worker_func, args=(evaluation_queue, result_queue, model.hr_t, model.tr_h)))
            workers[-1].start()
        # training iterations
        for n_iter in range(iter_offset, args.max_iter):
            start_time = timeit.default_timer()
            accu_loss = 0.
            accu_re_loss = 0.
            ninst = 0

            log.info("initializing raw training data...")
            nbatches_count = 0
            for dat in model.yeld_raw_training_data(batch_size=args.batch):
                raw_training_data_queue.put(dat)
                nbatches_count += 1
            log.info("raw training data initialized.")
            total_batch = nbatches_count  # batch size
            # log.debug("Stampa di conteggio ogni 100 elmenti:")
            while nbatches_count > 0:
                if nbatches_count % 100 == 0:
                    log.debug(f"{nbatches_count} da processare su un batch di {total_batch}")
                nbatches_count -= 1

                hr_tlist, hr_tweight, tr_hlist, tr_hweight = training_data_queue.get()

                l, rl, _ = session.run(
                    [train_loss, model.regularizer_loss, train_op], {train_hrt_input: hr_tlist,
                                                                     train_hrt_weight: hr_tweight,
                                                                     train_trh_input: tr_hlist,
                                                                     train_trh_weight: tr_hweight})

                # testing phase/session code starts here
                accu_loss += l
                accu_re_loss += rl
                ninst += (len(hr_tlist) + len(tr_hlist)) / 2

                if ninst % (5000) is not None:
                    log.debug(
                        '[%d sec](%d/%d) : %.2f -- loss : %.5f rloss: %.5f ' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            l / (len(hr_tlist) + len(tr_hlist)),
                            args.loss_weight * (rl / (len(hr_tlist) + len(tr_hlist)))))

            log.info(
                "iter %d avg loss %.5f, time %.3f" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

            if n_iter % args.save_per == 0 or n_iter == args.max_iter - 1:
                save_path = saver.save(session,
                                       os.path.join(args.save_dir,
                                                    "CrossE_" + str(args.prefix) + "_" + str(n_iter) + ".ckpt"))
                log.info("Model saved at %s" % save_path)
            # compute evaluation every 'eval_per' times or after the last training iteration
            if (n_iter % args.eval_per == 0 and n_iter != 0) or (iter_offset == args.max_iter - 1):
                compute_results(model, session, args.eval_batch, test_head, test_tail, test_input, evaluation_queue,
                                args.n_worker, result_queue, n_iter, args.max_iter, args.output_rank)
        # to evaluate after loading a model already fully trained
        if iter_offset == args.max_iter:
            compute_results(model, session, args.eval_batch, test_head, test_tail, test_input, evaluation_queue,
                            args.n_worker, result_queue, iter_offset - 1, args.max_iter, args.output_rank)

        # kill all the worker process
        num_worker = 0
        for p in workers:
            num_worker += 1
            p.terminate()
            # log.info('kill worker %d'%num_worker)

        # kill all the data generator procecss
        num_gengrator = 0
        for p in data_generators:
            num_gengrator += 1
            p.terminate()
            # log.info('kill data generator %d'%num_gengrator)

        # SAVING DATA FOR FURTHER EXPERIMENTS

        log.info("FINISHED~")
        ent_emb = model.ent_embedding.eval()
        rel_emb = model.rel_embedding.eval()
        inv_rel_emb = model.reverse_rel_embedding.eval()
        train_hr_t = model.train_hr_t
        train_tr_h = model.train_tr_h
        test_hr_t = model.test_hr_t
        test_tr_h = model.test_tr_h
        test_triples = model.test_triples
        var_to_save = [ent_emb, rel_emb, inv_rel_emb, train_hr_t, train_tr_h, test_hr_t, test_tr_h, test_triples]
        var_names = [nameof(ent_emb), nameof(rel_emb), nameof(inv_rel_emb), nameof(train_hr_t), nameof(train_tr_h), nameof(test_hr_t),
                     nameof(test_tr_h), nameof(test_triples)]

        for var, name in zip(var_to_save, var_names):
            # let's also serialize the objects
            filename = f"{out_save_path_p}{name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(var, f)
            """# let's save in human-readable formats
            if ("emb" in name) or ("triple" in name):
                # because these are lists and can be stored as dataframes and then CSVs
                data = pd.DataFrame(var)
                filename = f"{out_save_path_h}{name}.csv"
                data.to_csv(filename, header=False, index=False)
            else:
                filename = f"{out_save_path_h}{name}.txt"
                with open(filename, 'w') as f:
                    #dictionaries stored as jsons
                    #json.dump(var, f)
                    f.write(var)"""

        log.info(f"All data stored in {out_save_path_h}")

        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossE.')
    parser.add_argument('--methodname', dest='method_name', type=str, help='Method name', default='CrossE.py')
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./datasets/FB15k/')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=200)
    parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=4000)
    parser.add_argument("--comb", dest="combination_method", type=str, help="Combination method", default='simple')
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    parser.add_argument("--generator", dest='n_generator', type=int, help="Data generator", default=10)
    parser.add_argument("--eval_batch", dest="eval_batch", type=int, help="Evaluation batch size", default=500)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Model path", default='./save/')
    parser.add_argument("--load_model", dest='load_model', type=str, help="Model file", default=None)
    parser.add_argument("--save_per", dest='save_per', type=int, help="Save per x iteration", default=1000)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluate every x iteration", default=1)
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=100)
    parser.add_argument("--summary_dir", dest='summary_dir', type=str, help="summary directory",
                        default='./CrossE_summary/')
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",
                        default=0.5)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
    parser.add_argument("--prefix", dest='prefix', type=str, help="model_prefix", default='DEFAULT')
    parser.add_argument("--loss_weight", dest='loss_weight', type=float, help="Weight on parameter loss", default=1e-4)
    parser.add_argument("--neg_weight", dest='neg_weight', type=int, help="Sampling weight on negative examples",
                        default=50)
    parser.add_argument('--output_rank', dest='output_rank', type=bool, help='output the result rank or not',
                        default=False)
    parser.add_argument('--log_level', dest='log_level', type=str,
                        help='set the logging level, choose between info or debug',
                        default="info")

    args = parser.parse_args()
    # CSVs
    ######
    tf.compat.v1.disable_eager_execution()
    print("Avvio")
    # tf.app.run()
    model = main(args)

