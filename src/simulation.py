""" Utility functions for simulating data. """
import numpy as np
import sys
import pandas as pd
from collections import Counter, defaultdict
import re
from scipy.stats import zscore
import math
import itertools
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

import util




def estimate_propensities(T, C):
    # estimate treatment distribution for each strata of the confound 
    # directly from the data
    df = pd.DataFrame(zip(C, T), columns=['C', 'T'])
    T_levels = set(T)
    propensities = []
    for c_level in set(C):
        subset = df.loc[df.C == c_level]
        # NOTE: subset.T => transpose
        p_TgivenC = [
            float(len(subset.loc[subset['T'] == t])) / len(subset) 
            for t in T_levels
        ]
        propensities.append(p_TgivenC[1])

    return propensities 

# b0  makes treatment (thm?) sepearte more (i.e. give more 1's)
# b1 1, 10, 100, makes confound (buzzy/not) seperate more (drives means apart)
# gamma 0 , 1, 4, noise level
# offset moves propensities towards the middle so sigmoid can split them into some noise
def simulate_Y(C, T, b0=0.5, b1=10, gamma=0.0, offset=0.75):
    propensities = estimate_propensities(T, C)
    # propensities = [0.27, 0.7]
    out = []
    test = defaultdict(list)
    for Ci, Ti in zip(C, T):
        noise = np.random.normal(0, 1)
        y0 = b1 * (propensities[Ci] - offset)
        y1 = b0 + y0
        y = (1 - Ti) * y0 + Ti * y1 + gamma * noise # gamma
        simulated_prob = sigmoid(y)
        y0 = sigmoid(y0)
        y1 = sigmoid(y1)
        threshold = np.random.uniform(0, 1)
        Y = int(simulated_prob > threshold)
        out.append(Y)
        test[Ci, Ti].append(Y)

    return out


def adjust_propensity(df, target_propensities):
    # subset to to desired propensities (must be smaller than true)
    true_propensities = estimate_propensities(df.T_true, df.C_true)
    target_propensities = [0.90, 0.70] # works well

    for i, (pi_tgt, pi_true) in enumerate(zip(target_propensities, true_propensities)):
        # drop enough samples so that we get the desired propensity
        # inverse of y = x / (x + out sample size) gives you tgt number for proportion
        Ci_subset = df.loc[df.C_true == i]
        Ci_T0_subset = Ci_subset.loc[Ci_subset.T_true == 0]
        Ci_T1_subset = Ci_subset.loc[Ci_subset.T_true == 1]
        tgt_num = -len(Ci_T0_subset) * pi_tgt / (pi_tgt - 1)
        drop_prop = (len(Ci_T1_subset) - tgt_num) / len(Ci_T1_subset)
        df = df.drop(Ci_T1_subset.sample(frac=drop_prop).index)
    return df


def adjust_precision_recall(df, target_precision, target_recall):
    # now balance data again so proxy treatment has right precision
    x = defaultdict(int)
    for t_true, t_proxy in zip(df.T_true, df.T_proxy):
        x[t_true, t_proxy] += 1
    true_precision = x[1, 1] / (x[0, 1] + x[1, 1])
    true_recall = x[1, 1] / (x[1, 1] + x[1, 0])

    true1_subset = df.loc[df.T_true == 1]
    true0_subset = df.loc[df.T_true == 0]
    true1_proxy1_subset = true1_subset.loc[true1_subset.T_proxy == 1]
    true1_proxy0_subset = true1_subset.loc[true1_subset.T_proxy == 0]
    true0_proxy1_subset = true0_subset.loc[true0_subset.T_proxy == 1]

    if target_precision > true_precision:
        # adjust precision with inverse of y = tp / (tp + x)
        tgt_num_t0p1 = -len(true1_proxy1_subset) * (target_precision - 1) / target_precision
        drop_prop = (len(true0_proxy1_subset) - tgt_num_t0p1) / len(true0_proxy1_subset)
        df = df.drop(true0_proxy1_subset.sample(frac=drop_prop).index)
    else:
        # adjust down with inverse of y = x / (x + fp)
        tgt_num_t1p1 = - (len(true0_proxy1_subset) * target_precision) / (target_precision - 1)
        drop_prop = (len(true1_proxy1_subset) - tgt_num_t1p1) / len(true1_proxy1_subset)
        df = df.drop(true1_proxy1_subset.sample(frac=drop_prop).index)

    # refresh subsets (TODO refactor)
    true1_subset = df.loc[df.T_true == 1]
    true0_subset = df.loc[df.T_true == 0]
    true1_proxy1_subset = true1_subset.loc[true1_subset.T_proxy == 1]
    true1_proxy0_subset = true1_subset.loc[true1_subset.T_proxy == 0]
    true0_proxy1_subset = true0_subset.loc[true0_subset.T_proxy == 1]

    if target_recall > true_recall:
        # adjust recall with inverse of t1p1 / (t1p1 + x)
        tgt_num_t1p0 = -len(true1_proxy1_subset) * (target_recall - 1) / target_recall
        drop_prop = (len(true1_proxy0_subset) - tgt_num_t1p0) / len(true1_proxy0_subset)
        df = df.drop(true1_proxy0_subset.sample(frac=drop_prop).index)
    else:
        # adjust down with inverse of y = x / (x + fn)
        tgt_num_t1p1 = - (len(true1_proxy0_subset) * target_recall) / (target_recall - 1)
        drop_prop = (len(true1_proxy1_subset) - tgt_num_t1p1) / len(true1_proxy1_subset)
        df = df.drop(true1_proxy1_subset.sample(frac=drop_prop).index)

    return df


def adjust_proxy_propensities(df, target_t0, target_t1):
    # adjust P(T* = 1 | T=?)
    x = defaultdict(int)
    t0_subset = df.loc[df.T_true == 0]
    t1_subset = df.loc[df.T_true == 1]
    tp1_t0_subset = t0_subset.loc[df.T_proxy == 1]
    tp1_t1_subset = t1_subset.loc[df.T_proxy == 1]
    true_t0 = len(tp1_t0_subset) / len(t0_subset)
    true_t1 = len(tp1_t1_subset) / len(t1_subset)

    # (1) adjust P(T*=1 | T=0)
    # drop from numerator
    if target_t0 < true_t0:
        # inverse of (n - x)/(d - x) = y
        #   is (dy - n) / (y - 1)
        drop_num = (len(t0_subset) * target_t0 - len(tp1_t0_subset)) / (target_t0 - 1)
        drop_prop = drop_num / len(tp1_t0_subset)
        df = df.drop(tp1_t0_subset.sample(frac=drop_prop).index)
    # drop from denominator (easier b/c can just drop from other part of t0_subset)
    elif target_t0 > true_t0:
        drop_num = len(t0_subset) - (len(tp1_t0_subset) / target_t0)
        tp0_t0_subset = t0_subset.loc[df.T_proxy == 0]
        drop_prop = drop_num / len(tp0_t0_subset)
        df = df.drop(tp0_t0_subset.sample(frac=drop_prop).index)

    # refresh subsets (TODO refactor)
    t0_subset = df.loc[df.T_true == 0]
    t1_subset = df.loc[df.T_true == 1]
    tp1_t0_subset = t0_subset.loc[df.T_proxy == 1]
    tp1_t1_subset = t1_subset.loc[df.T_proxy == 1]
    true_t0 = len(tp1_t0_subset) / len(t0_subset)
    true_t1 = len(tp1_t1_subset) / len(t1_subset)

    # (1) adjust P(T*=1 | T=1)
    # drop from numerator
    if target_t1 < true_t1:
        # inverse of (n - x)/(d - x) = y
        #   is (dy - n) / (y - 1)
        drop_num = (len(t1_subset) * target_t1 - len(tp1_t1_subset)) / (target_t1 - 1)
        drop_prop = drop_num / len(tp1_t1_subset)
        df = df.drop(tp1_t1_subset.sample(frac=drop_prop).index)
    # drop from denominator (easier b/c can just drop from other part of t0_subset)
    elif target_t1 > true_t1:
        drop_num = len(t1_subset) - (len(tp1_t1_subset) / target_t1)
        tp0_t1_subset = t1_subset.loc[df.T_proxy == 0]
        drop_prop = drop_num / len(tp0_t1_subset)
        df = df.drop(tp0_t1_subset.sample(frac=drop_prop).index)

    return df


def adjust_accuracy(df, target_acc):
    true_acc = accuracy_score(df.T_true, df.T_proxy)
    rights = df.loc[df.T_proxy == df.T_true]
    wrongs = df.loc[df.T_proxy != df.T_true]
    if true_acc < target_acc:
        # adjust up by 
        drop_num = len(wrongs) - (len(rights) - target_acc * len(rights)) / target_acc
        drop_prop = drop_num / len(wrongs)
        df = df.drop(wrongs.sample(frac=drop_prop).index)
    else:
        drop_num = len(rights) - len(wrongs) * len(rights) / (len(rights) - 1)
        drop_prop = drop_num / len(rights)
        df = df.drop(rights.sample(frac=drop_prop).index)

    return df


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def run_simulation(
    df,
    propensities=[0.9, 0.7],
    precision=0.9,
    recall=0.95,
    b0=0.5,
    b1=10,
    gamma=0.0,
    offset=0.75,
    accuracy=-1,
    proxy_type='lex',
    size=-1):
    """Use the star rating of a music review for T
        it's sentiment/bert predictions/random sampling for T*
        CD/MP3 for C
        and then simulate Y ~ f(T, C)
    """
    # Get T from the ground truth rating
    def treatment_from_rating(rating):
        return int(rating == 5.0)

    df['T_true'] = df['rating'].apply(treatment_from_rating)

    # Get T* from a sentiment lexicon
    if proxy_type == 'lex':
        lex = util.read_lexicon('./positive-words.txt')

        def proxy_treatment_from_review(text):
            text = util.word_tokenize(text.lower())
            return int(len(set(text) & lex) > 0)

        df['T_proxy'] = df['text'].apply(proxy_treatment_from_review)

    # Get a randomly sampled proxy
    elif proxy_type == 'random':
        randoms = np.random.uniform(0, 1, len(df['T_true']))
        if isinstance(accuracy, tuple): # TODO this is a hack
            pThatGivenT = accuracy
        elif accuracy > 0:
            pThatGivenT = [1 - accuracy, accuracy]
        else:
            pThatGivenT = [0.2, 0.8]
        mask = np.array([pThatGivenT[ti] for ti in df['T_true']])
        df['T_proxy'] = (randoms < mask).astype(int)

    # get confound (lump together mp3 + vinyl)
    C_from_product = lambda p: 1 if p == 'audio cd' else 0
    df['C_true'] = df['product'].apply(C_from_product)

    df = adjust_propensity(df, propensities)

    df['Y_sim'] = simulate_Y(df.C_true, df.T_true,
        b0=b0, b1=b1, gamma=gamma, offset=offset)

    # optionally adjust acc, pre, rec after you've drawn Y's
    if proxy_type != 'random' and (isinstance(accuracy, tuple) or accuracy > 0):
        df = adjust_accuracy(df, accuracy)
    elif precision > 0 and recall > 0:
        df = adjust_precision_recall(df, precision, recall)

    if size > 0:
        sample_size = float(size) / len(df)
        df = df.sample(frac=sample_size)

    # reindex to rm phantom rows
    df = df.reset_index()

    return df
