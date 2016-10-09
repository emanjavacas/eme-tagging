
import sys

try:
    from .config import penn_root as root
except:
    raise Warning("set global variable ``penn_root'' to the appropriate value")

import os
import re
import glob
import random


genre_mapping = {
    '"BIBLE"': '"RELIGION"',
    '"BIOGRAPHY_AUTO"': '"DIARY"',
    '"BIOGRAPHY_LIFE_OF_SAINT"': '"NARRATIVE"',
    '"BIOGRAPHY_OTHER"': '"NARRATIVE"',
    '"DIARY_PRIV"': '"DIARY"',
    '"DRAMA_COMEDY"': '"FICTION"',
    '"EDUC_TREATISE"': '"NONFICTION"',
    '"FICTION"': '"FICTION"',
    '"HANDBOOK_ASTRO"': '"NONFICTION"',
    '"HANDBOOK_MEDICINE"': '"NONFICTION"',
    '"HANDBOOK_OTHER"': '"NONFICTION"',
    '"HISTORY"': '"NARRATIVE"',
    '"HOMILY"': '"RELIGION"',
    '"HOMILY_POETRY"': '"RELIGION"',
    '"LAW"': '"NONFICTION"',
    '"LETTERS_NON-PRIV"': '"LETTERS"',
    '"LETTERS_PRIV"': '"LETTERS"',
    '"PHILOSOPHY"': '"NARRATIVE"',
    '"PHILOSOPHY/FICTION"': '"NARRATIVE"',
    '"PROCEEDINGS_TRIAL"': '"NONFICTION"',
    '"RELIG_TREATISE"': '"RELIGION"',
    '"ROMANCE"': '"FICTION"',
    '"RULE"': '"NONFICTION"',
    '"SCIENCE_MEDICINE"': '"NONFICTION"',
    '"SCIENCE_OTHER"': '"NONFICTION"',
    '"SERMON"': '"RELIGION"',
    '"TRAVELOGUE"': '"DIARY"'
}

INF = float('inf')


def intersection(*seqs):
    a, rest = list(seqs).pop(), seqs
    return [set(a) & set(s) for s in rest][0]


def take(g, n):
    index = 0
    for x in g:
        if index >= n:
            break
        yield x
        index += 1


def shuffle_seq(seq):
    return sorted(seq, key=lambda k: random.random())


def split_sents(sents):
    """
    Example:
    X_train, y_train = split_sents(pos.pos_from_range(1600, 1650))
    """
    X, y = [], []
    for sent in sents:
        words, tags = zip(*sent)
        X.extend(words)
        y.extend(tags)
    return X, y


def read_year(fname):
    fname = os.path.basename(fname)
    return re.findall(r'\d+', fname)[0]


def read_info(in_fn=os.path.join(root, "corpus_data.csv")):
    result = {}
    with open(in_fn, "r") as f:
        next(f)
        for l in f:
            l = l.strip()
            fname, date, genre, wc, genre2, span = l.split(",")
            result[fname.strip('"')] = tuple([date, genre, wc, genre2, span])
    return result


def abspath(basename, ext):
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*." + ext)
    fname = ".".join([basename, ext])
    for f in fnames:
        if fname in f:
            return f


def files_in_range(from_y, to_y, ext='pos'):
    info = read_info()
    result = []
    for f, row in info.items():
        year = int(row[0].split('-')[0])
        if from_y <= year < to_y:
            result.append(abspath(f, ext))
    return result
