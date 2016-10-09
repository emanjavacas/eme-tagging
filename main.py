
import sys
import random
import itertools
import json
from collections import Counter
import numpy as np

import fasttext
from keras.utils.np_utils import to_categorical

from casket import Experiment as E
from casket.nlp_utils import Indexer
from casket.nlp_utils.corpus import pad
from rnn_tagger import bilstm_tagger
from data.Penn.pos import pos_from_range


BATCH_MSG = "Epoch: %2d, Batch: %4d, Loss: %.4f, Dev-loss: %.4f: Dev-acc: %.4f"
EPOCH_MSG = "\nEpoch: %2d, Loss: %.4f, Dev-loss: %.4f: Dev-acc: %.4f\n"


def log_batch(epoch, batch, train_loss, dev_loss, dev_acc):
    print(BATCH_MSG % (epoch, batch, train_loss, dev_loss, dev_acc), end='\r')


def log_epoch(epoch, train_loss, dev_loss):
    print(EPOCH_MSG % (epoch, train_loss, dev_loss, dev_acc))


def dump_json(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f)


def pos_hash(d):    
    hash(frozenset(d)) % ((sys.maxsize + 1) * 2)


def pos_counts(from_y, to_y):
    tags = Counter()
    maxlen = 0
    for sent in pos_from_range(from_y, to_y):
        for _, pos in sent:
            tags[pos] += 1
        maxlen = max(maxlen, len(sent))
    return maxlen, tags


def to_array(sent, idxr, ft, maxlen):
    zeros = np.zeros(ft.dim, dtype=np.float32)
    words, pos = zip(*sent)
    words = [ft[w] for w in words]
    X = pad(list(words), maxlen, paditem=zeros)
    pos = pad(idxr.transform(pos), maxlen, paditem=0)
    y = to_categorical(pos, nb_classes=idxr.vocab_len())
    return X, y


def batches(sents, idxr, ft, maxlen, batch_size=500):
    c = 0
    X_batch, y_batch = [], []
    for sent in sents:
        c += 1
        X, y = to_array(sent, idxr, ft, maxlen)
        X_batch.append(X), y_batch.append(y)
        if c % batch_size == 0:
            yield np.asarray(X_batch), np.asarray(y_batch)
            X_batch, y_batch = [], []


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--years_range', nargs='+', type=int,
                        default=(1500, 1800))
    parser.add_argument('-s', '--seed', default=1001, type=int)
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop')
    parser.add_argument('-r', '--rnn_layers', type=int, default=1)
    parser.add_argument('-l', '--lstm_dim', type=int, default=100)
    parser.add_argument('-H', '--hidden_dim', type=int, default=250)
    parser.add_argument('-D', '--dropout', type=float, default=0.0)
    parser.add_argument('-f', '--fasttext_model', type=str,
                        default='data/fastText/tcp_full.bin')
    parser.add_argument('-b', '--batch_size', type=int, default=50)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-L', '--loss', type=int, default=1,
                        help='report loss every L batches')
    args = parser.parse_args()

    # params
    FROM_Y, TO_Y = args.years_range
    OPTIMIZER = args.optimizer
    RNN_LAYERS = args.rnn_layers
    LSTM_DIM = args.lstm_dim
    HIDDEN_DIM = args.hidden_dim
    DROPOUT = args.dropout
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    SEED = args.seed

    # load tools
    ft = fasttext.load_model(args.fasttext_model)
    maxlen, tags = pos_counts(FROM_Y, TO_Y)
    total_counts = sum(tags.values())
    idxr = Indexer(pad='PAD', oov=None)
    idxr.fit(tags.keys())
    class_weight = {idxr.encode(t): c/total_counts for (t, c) in tags.items()}

    print("Loading datasets")
    random.seed(SEED)
    sents = pos_from_range(FROM_Y, TO_Y, shuffle=True)
    test = itertools.islice(sents, 1000)
    dev = itertools.islice(sents, 1000)
    X_test, y_test = list(zip(*[to_array(s, idxr, ft, maxlen) for s in test]))
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    X_dev, y_dev = list(zip(*[to_array(s, idxr, ft, maxlen) for s in dev]))
    X_dev, y_dev = np.asarray(X_dev), np.asarray(y_dev)

    print("Compiling model")
    tagger = bilstm_tagger(ft, idxr.vocab_len(), maxlen,
                           lstm_dims=LSTM_DIM, hidden_dim=HIDDEN_DIM,
                           rnn_layers=RNN_LAYERS, dropout=DROPOUT)
    tagger.compile(OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    # model params
    params = {'rnn_layers': RNN_LAYERS, 'lstm_layer': LSTM_DIM,
              'hidden_layer': HIDDEN_DIM, 'optimizer': OPTIMIZER,
              'batch_size': BATCH_SIZE, 'dropout': DROPOUT,
              'pos_tags': list(tags.keys()), 'ft_model': args.fasttext_model,
              'range': [FROM_Y, TO_Y], 'seed': SEED}
    model_hash = pos_hash(params)

    print("Training")
    model = E.use(args.db, exp_id='rnn_tagger').model('bilstm')
    with model.session(params) as session:
        from time import time
        start = time()        
        for e in range(args.epochs):
            losses = []
            bs = batches(sents, idxr, ft, maxlen, batch_size=BATCH_SIZE)
            for b, (X, y) in enumerate(bs):
                loss, _ = tagger.train_on_batch(X, y#, class_weight=class_weight
                )
                losses.append(loss)
                if b % args.loss == 0:
                    dev_loss, dev_acc = tagger.test_on_batch(X_dev, y_dev)
                    log_batch(e, b, np.mean(losses), dev_loss, dev_acc)
            log_epoch(e, np.mean(losses), dev_loss, dev_acc)                    
            session.add_epoch(
                e, {'training_loss': str(np.mean(losses)),
                    'dev_loss': str(dev_loss),
                    'dev_acc': str(dev_acc)})
        _, test_acc = tagger.test_on_batch(X_test, y_test)
        print("Test acc: %.4f\n" % test_acc)
        session.add_result({'test_acc': str(test_acc)})
        session.add_meta({'run_time': time() - start,
                          'model_prefix': model_hash})

    tagger.save_weights('./models/' + str(model_hash) + '_weights.h5')
    dump_json(tagger.get_config(), './models/' + str(model_hash) + '_config.json')
    idxr.save('./models/' + str(model_hash) + '_indexer.json')
