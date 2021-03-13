import argparse
import math
import os
import os.path
from global_logger import Log
import pickle
import timeit
from multiprocessing import JoinableQueue, Queue, Process

from pathlib import Path
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

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
    def ent_embedding(self):
        """
        Used to contain the embeddings of the entities
        :return:
        """
        return self.__ent_embedding

    @property
    def rel_embedding(self):
        return self.__rel_embedding

    def raw_training_data(self, batch_size=100):
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

    def testing_data(self, batch_size=100):
        n_triple = len(self.__test_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__test_triple[start:end, :]
            start = end

    def valid_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end

    def __init__(self, data_dir, embed_dim=100, combination_method='simple', dropout=0.5, neg_weight=0.5):
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
            # TODO: controllare, potrebbe essere l'operatore di score
            hrt_res = tf.matmul(tf.nn.dropout(tf.tanh(hr_tlist_hr + self.__hr_combination_bias), self.__dropout),
                                self.__ent_embedding,
                                transpose_b=True)

            tr_hlist_tr = tr_hlist_t * t_weight + tr_hlist_r * tr_hlist_t * t_weight

            trh_res = tf.matmul(tf.nn.dropout(tf.tanh(tr_hlist_tr + self.__tr_combination_bias), self.__dropout),
                                self.__ent_embedding,
                                transpose_b=True)

            regularizer_loss = 0.0
            for param in self.__trainable:
                regularizer_loss += tf.reduce_sum(tf.abs(param))
            self.regularizer_loss = regularizer_loss

            hrt_res_sigmoid = tf.sigmoid(hrt_res)

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
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            h = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 0])
            t = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 1])
            r = tf.nn.embedding_lookup(self.__rel_embedding, inputs[:, 2])
            r_reverse = tf.nn.embedding_lookup(self.__rel_embedding_reverse, inputs[:, 2])
            h_w = tf.nn.embedding_lookup(self.__h_weighted_vector, inputs[:, 2])
            t_w = tf.nn.embedding_lookup(self.__t_weighted_vector, inputs[:, 2])

            ent_mat = tf.transpose(self.__ent_embedding)

            # predict tails
            hr = h * h_w + r * h * h_w

            hrt_res = tf.sigmoid(tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat))
            _, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)

            # predict heads
            tr = t * t_w + r_reverse * t * t_w

            trh_res = tf.sigmoid(tf.matmul(tf.tanh(tr + self.__tr_combination_bias), ent_mat))
            _, head_ids = tf.nn.top_k(trh_res, k=self.__n_entity)

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
    Uses the data contained in the in_queue to run evaluation and put the results into the out_queue
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
        for val in head_pred[i]:
            if val == h:
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


def main(_):
    parser = argparse.ArgumentParser(description='CrossE.')
    parser.add_argument('--methodname', dest='method_name', type=str, help='Method name', default='CrossE.py')
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./datasets/FB15k/')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=200)
    parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=4000)
    parser.add_argument("--comb", dest="combination_method", type=str, help="Combination method", default='simple')
    # TODO: cosa sono i workers?
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    # TODO: cosa è il generator?
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
    parser.add_argument('--log_level', dest='log_level', type=str, help='set the logging level, choose between info or debug',
                        default="info")

    args = parser.parse_args()
    # creating the folder for logs
    log_save_dir = f"{args.save_dir}execution_logs"  # to save a subfolder with the fraction used
    Path(log_save_dir).mkdir(parents=True, exist_ok=True)
    if args.log_level == "debug":
        log = Log.get_logger(logs_dir=log_save_dir, level=Log.Levels.DEBUG)
    else:
        log = Log.get_logger(logs_dir=log_save_dir)

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
        # TODO: capire come accedere ai tensori
        if args.load_model is not None:
            saver.restore(session, args.load_model)
            # extracts, from the checkpoint filename, the iteration number to continue from
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
        result_queue = Queue()
        workers = list()
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
            for dat in model.raw_training_data(batch_size=args.batch):
                raw_training_data_queue.put(dat)
                nbatches_count += 1
            log.info("raw training data initialized.")
            total_batch = nbatches_count  # batch size
            log.debug("Stampa di conteggio ogni 100 elmenti:")
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

                accu_loss += l
                accu_re_loss += rl
                ninst += (len(hr_tlist) + len(tr_hlist)) / 2

                if ninst % (5000) is not None:
                    log.debug(
                        '[%d sec](%d/%d) : %.2f -- loss : %.5f rloss: %.5f ' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            l / (len(hr_tlist) + len(tr_hlist)),
                            args.loss_weight * (rl / (len(hr_tlist) + len(tr_hlist)))))

            # log.info("")
            log.info(
                "iter %d avg loss %.5f, time %.3f" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

            if n_iter % args.save_per == 0 or n_iter == args.max_iter - 1:
                save_path = saver.save(session,
                                       os.path.join(args.save_dir,
                                                    "CrossE_" + str(args.prefix) + "_" + str(n_iter) + ".ckpt"))
                log.info("Model saved at %s" % save_path)

            if (n_iter % args.eval_per == 0 and n_iter != 0) or n_iter == args.max_iter - 1:
                for data_func, test_type in zip([model.valid_data, model.testing_data], ['valid', 'TEST']):
                    accu_mean_rank_h = list()
                    accu_mean_rank_t = list()
                    accu_mrr_h = list()
                    accu_mrr_t = list()
                    accu_filtered_mean_rank_h = list()
                    accu_filtered_mean_rank_t = list()
                    accu_fmrr_h = list()
                    accu_fmrr_t = list()

                    evaluation_count = 0

                    for testing_data in data_func(batch_size=args.eval_batch):
                        head_pred, tail_pred = session.run([test_head, test_tail],
                                                           {test_input: testing_data})

                        evaluation_queue.put((testing_data, head_pred, tail_pred))
                        evaluation_count += 1

                    for i in range(args.n_worker):
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

                    if n_iter == args.max_iter - 1 and args.output_rank == True:
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

        log.info("FINISHED~")


tf.compat.v1.disable_eager_execution()
if __name__ == '__main__':
    tf.app.run()
