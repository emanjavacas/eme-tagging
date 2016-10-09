


def tree_from_file(fname):
    with codecs.open(fname, "r", "utf-8") as f:
        tree_string = ""
        for l in f:
            l = l.strip()
            if not l:
                yield parse(tree_string)
                tree_string = ""
            else:
                tree_string += l


def get_pos(simple_tags=True):
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*.pos")
    for f in fnames:
        for pos in pos_from_file(f, simple_tags=simple_tags):
            yield pos


# kindly taken from http://norvig.com/lispy.html
def tokenize(tree_string):
    return tree_string.replace('(', ' ( ').replace(')', ' ) ').split()


def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        return L
    elif token == ')':
        raise SyntaxError("unexpected )")
    else:
        return atom(token)


def atom(token):
    return token


def parse(string):
    return read_from_tokens(tokenize(string))


def get_psd():
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*.psd")
    for f in fnames:
        for tree in tree_from_file(f):
            yield tree
