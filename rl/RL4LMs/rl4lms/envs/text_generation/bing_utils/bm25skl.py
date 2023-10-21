""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1



#------------ End of library impl. Followings are the example -----------------

# from sklearn.datasets import fetch_20newsgroups


# texts = fetch_20newsgroups(subset='train').data
# bm25 = BM25()
# docs = ['specific prompt formats that work particularly well',
#         'Due to the way the instruction-following models are trained or the data they are trained on, there are specific prompt formats that work particularly well and align better with the tasks at hand. Below we present a number of prompt formats we find work reliably well, but feel free to explore different formats, which may fit your task best.',
#         'For best results, we generally recommend using the latest, most capable models. As of November 2022, the best options are the “text-davinci-003” model for text generation, and the “code-davinci-002” model for code generation.',
#         'For best results, we generally recommend using the latest, most capable models. As of November 2022, the best options are the “text-davinci-003” model for text generation, and the “code-davinci-002” model for code generation.']
# bm25.fit(docs)
# print(bm25.transform('specific prompt formats that work particularly well', docs))

def bm25score(docs, q, max_words, topp, use):
    # print('bm25: ', use)
    bm25f = BM25()
    try:
        bm25f.fit(docs)
    except ValueError:
        return ''
    scores = list(bm25f.transform(q,docs))
    # print('scores: ', scores)
    ind = []
    doc = ''
    if use == 'words':
        for _ in range(len(scores)):
            indi = scores.index(max(scores))
            ind.append(indi)
            if max(scores) == 0:
                break
            if max_words>0 and len(doc.split(" ")) + len(docs[indi].split(" ")) > max_words:
                break
            doc += docs[indi]
            scores[indi] = -1
    # use topp
    elif use == 'topp':
        thre = max(scores) * (1 - topp)
        sele_id = [ 1 if s > thre else 0 for s in scores]
        for i in range(len(scores)):
            if scores[i] > thre:
                doc += docs[i].strip()
    
    return doc

    

# def bm25gen()