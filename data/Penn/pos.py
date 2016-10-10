
import re
import itertools

from .utils import INF, files_in_range, shuffle_seq


def simplify_tag(tag):
    if '+' in tag:
        tag = tag.split('+')[0]
    tag = re.sub(r'[0-9]+$', '', tag)
    return tag


def pos_from_file(fname, rem_id=True, simple_tags=True):
    with open(fname, "r") as f:
        sent = []
        for l in f:
            l = l.strip()
            if l.startswith("<") or l.startswith("{"):  # ignore metadata
                continue
            if not l and not sent:  # blank lines
                continue
            elif not l:
                yield sent
                sent = []
                continue
            word, tag = l.split("/")
            if tag == 'ID' and rem_id:
                continue
            tag = simplify_tag(tag) if simple_tags else tag
            sent.append((word, tag))


def pos_from_files(files, max_sents=INF, maxlen=INF, rem_id=True, shuffle=False):
    sents = (sent for f in files for sent in pos_from_file(f, rem_id=rem_id)
             if len(sent) <= maxlen)
    if shuffle:
        return itertools.islice(shuffle_seq(sents), max_sents)
    return itertools.islice(sents, max_sents)


def pos_from_range(from_y, to_y, max_sents=INF, maxlen=INF,
                   rem_id=True, shuffle=False):
    files = files_in_range(from_y, to_y)
    return pos_from_files(files, max_sents=max_sents, maxlen=maxlen,
                          rem_id=rem_id, shuffle=shuffle)

