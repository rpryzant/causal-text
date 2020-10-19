"""

"""

from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random

import torch

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn import feature_extraction

import simulation
import label_expansion
import util

import CausalBert

def prepare_covariates(df, 
    stopwords=None,
    vocab_size=2000,
    use_counts=False):

    def admissable(w):
        if stopwords is None:
            return True
        return w not in stopwords

    # 2k most common not in lex
    c = Counter([w for s in df['text'] for w in util.word_tokenize(s.lower()) if admissable(w)])
    vocab = list(zip(*c.most_common(vocab_size)))[0]

    # vectorize inputs
    vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=util.word_tokenize,
        vocabulary=vocab,
        binary=(not use_counts),
        ngram_range=(1, 1))
    corpus = list(df['text'])
    vectorizer.fit(corpus)
    X = vectorizer.transform(corpus).todense()
    return X, vocab, vectorizer


def run_parameterized_estimators(
    df, df2=None,
    stopwords=None,
    vocab_size=2000,
    use_counts=False,
    threshold=0.8,
    only_zeros=True,
    inner_alpha='optimal',
    outer_alpha='optimal',
    g_weight=1, Q_weight=1, mlm_weight=1, run_cb=False):
    """ Run all the ATE estimators based on models:
            regression expansion (+pu classifier), bert adjustment, and
                regression expansion + bert.
    """
    X, vocab, vectorizer = prepare_covariates(df, stopwords, vocab_size, use_counts)
    T_true = df['T_true'].to_numpy()
    T_proxy = df['T_proxy'].to_numpy()

    # PU classifier expansion
    only_zeros=True
    pu = label_expansion.PUClassifier(
        inner_alpha=inner_alpha,
        outer_alpha=outer_alpha)
    pu.fit(X, T_proxy)
    T_plus_pu = label_expansion.expand_variable(pu, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)
    ATE_pu = util.ATE_adjusted(df.C_true, T_plus_pu , df.Y_sim)

    # Plain regression expansion
    reg = SGDClassifier(loss="log", penalty="l2", alpha=outer_alpha)
    reg.fit(X, T_proxy)
    T_plus_reg = label_expansion.expand_variable(reg, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)
    ATE_reg = util.ATE_adjusted(df.C_true, T_plus_reg , df.Y_sim)

    if run_cb:
        cbw = CausalBert.CausalBertWrapper(g_weight=g_weight, Q_weight=Q_weight, mlm_weight=mlm_weight)
        cbw.train(df['text'], df.C_true, df.T_proxy, df.Y_sim, epochs=3)
        ATE_cb_Tproxy = cbw.ATE(df.C_true, df['text'], Y=df.Y_sim, platt_scaling=False)

        cbw = CausalBert.CausalBertWrapper(g_weight=g_weight, Q_weight=Q_weight, mlm_weight=mlm_weight)
        cbw.train(df['text'], df.C_true, T_plus_pu, df.Y_sim, epochs=3)
        ATE_cb_Tplus = cbw.ATE(df.C_true, df['text'], Y=df.Y_sim, platt_scaling=False)

    else:
        ATE_cb_Tproxy, ATE_cb_Tplus = -1, -1

    return ATE_pu, ATE_reg, ATE_cb_Tproxy, ATE_cb_Tplus


def get_data(args):
    """ Read in a dataset and make sure it has fields
            text, T_true, T_proxy, C_true, Y_sim
    """
    if args.data_type == 'music':
        # Add columns T_true T_proxy C_true Y_sim to the data
        df = pd.read_csv(args.data, sep='\t', error_bad_lines=False)
        df['text'] = df['text'].map(lambda x: x.lower() if isinstance(x,str) else x)
        df = simulation.run_simulation(df,
            propensities=[args.p1, args.p2] if args.p1 > 0 else None,
            precision=args.pre,
            recall=args.rec,
            b0=args.b0,
            b1=args.b1,
            offset=args.off,
            gamma=args.gamma,
            accuracy=args.acc,
            proxy_type=args.ptype,
            size=args.size)
    elif args.data_type == 'custom':
        # use what's given without any changes
        # (T_true, T_proxy, C_true, and Y should already be in there)
        df = pd.read_csv(args.data, sep='\t', error_bad_lines=False)
        df['text'] = df['text'].map(lambda x: x.lower() if isinstance(x,str) else x)
        df['Y_sim'] = df['Y']
    else:
        raise Exception('Unknown data type: %s' % args.data_type)

    return df


def run_experiment(args):
    """ Run an experiment with the given args and seed.

        Returns {causal estimator: ATE estimate}
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    df = get_data(args)

    unadjusted = util.ATE_unadjusted(df.T_true, df.Y_sim)
    adjusted = util.ATE_adjusted(df.C_true, df.T_true, df.Y_sim)

    unadjusted_proxy = util.ATE_unadjusted(df.T_proxy, df.Y_sim)
    adjusted_proxy = util.ATE_adjusted(df.C_true, df.T_proxy, df.Y_sim)

    ATE_matrix = util.ATE_matrix(df.T_true, df.T_proxy, df.C_true, df.Y_sim)

    ATE_pu, ATE_reg, ATE_cb_Tproxy, ATE_cb_Tplus = run_parameterized_estimators(
        df=df,
        vocab_size=args.vs,
        threshold=args.thre,
        inner_alpha=args.ina,
        outer_alpha=args.outa, 
        g_weight=args.g_weight, 
        Q_weight=args.Q_weight, 
        mlm_weight=args.mlm_weight, 
        run_cb=args.run_cb)

    return {
        'ate_T': adjusted,
        'ate_matrix': ATE_matrix,
        'unadj_That': unadjusted_proxy,
        'ate_That': adjusted_proxy,
        'ate_T+_pu': ATE_pu,
        'ate_T+_reg': ATE_reg,
        'ate_That_cb': ATE_cb_Tproxy,
        'ate_T+_cb': ATE_cb_Tplus,
        'unadj_T': unadjusted,
    }

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--p1', type=float, help='P(T* = 0 | C) in simulation (-1 to ignore)')
    parser.add_argument('--p2', type=float, help='P(T* = 1 | C) in simulation (-1 to ignore)')
    parser.add_argument('--ptype', type=str, help='type of T*', choices=['random', 'bert', 'lex'])
    parser.add_argument('--acc', type=float, help='T*/T accuracy (-1 to ignore).')
    parser.add_argument('--pre', type=float, help='Precision between T* and T (-1 to ignore)')
    parser.add_argument('--rec', type=float, help='Recall between T* and T (-1 to ignore)')
    parser.add_argument('--b0', type=float, help='Simulated treatment strength')
    parser.add_argument('--b1', type=float, help='Simulated confound strength')
    parser.add_argument('--gamma', type=float, help='Noise level in simulation')
    parser.add_argument('--off', type=float, help='Simulated offset for T/C pre-threshold means')
    parser.add_argument('--size', type=str, help='Sample size if you want to sub-sample the data (-1 to ignore)')
    parser.add_argument('--vs', type=int, help='Vocab size for T+ model')
    parser.add_argument('--ina', type=float, help='One-class regression inner alpha (regularization strength)')
    parser.add_argument('--outa', type=float, help='One-class regression outer alpha (regularization strength)')
    parser.add_argument('--thre', type=float, help='T+ classifier threshold')
    parser.add_argument('--g_weight', type=float, help='Loss weight for the g head in Causal Bert.')
    parser.add_argument('--Q_weight', type=float, help='Loss weight for the Q head in Causal Bert.')
    parser.add_argument('--mlm_weight', type=float, help='Loss weight for the mlm head in Causal Bert.')
    parser.add_argument('--run_cb', type=bool, help='Whether to run causal bert or not.')
    parser.add_argument('--data_path', type=str, help='name of csv to slurp in. ')
    parser.add_argument('--data_type', type=str, help='one of [music, custom]')
    parser.add_argument('--o', type=str, help='output filename')

    parser.set_defaults(
        p1=0.9, # 0.88
        p2=0.7, # 0.842
        pre=-1, # 0.94
        rec=-1, # 0.98
        acc=-1,
        b0=0.8, # 0.4, 0.8
        b1=4.0, # -0.4, 0.4
        gamma=1.0, # 0, 1
        off=0.9,   
        size=-1,
        seed=420,
        vs=2000,
        ina=0.00359,
        outa=0.00077,
        thre=0.22,
        g_weight=0.0,
        Q_weight=0.1,
        mlm_weight=1.0,
        run_cb=False,
        ptype='random',
        data='./music.tsv',
        data_type='music',
        o='./out.txt'
    )
    args = parser.parse_args()

    results = run_experiment(args)
    out = {**vars(args), **results}
    print(out)

    quit()
