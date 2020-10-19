""" Msc utility functions, mostly for computing statistics from the data."""
# try:
#   - more estimators (https://arxiv.org/pdf/1702.01250.pdf)
#   - inverse propensity score (lecture5.pdf)
#   - augmented inverse propensity score (lecture5.pdf)


from collections import defaultdict
import re

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

def word_tokenize(s):
    return re.findall('\w+', s)


def read_lexicon(f):
    lex = {
        l.strip().lower() for l in open(f)
        if not l.startswith(';') and l.strip()
    }
    return lex


def ATE_unadjusted(T, Y):
    x = defaultdict(list)
    for t, y in zip(T, Y):
        x[t].append(y)
    T0 = np.mean(x[0])
    T1 = np.mean(x[1])
    return T0 - T1

def ATE_adjusted(C, T, Y):
    x = defaultdict(list)
    for c, t, y in zip(C, T, Y):
        x[c, t].append(y)

    C0_ATE = np.mean(x[0,0]) - np.mean(x[0,1])
    C1_ATE = np.mean(x[1,0]) - np.mean(x[1,1])
    return np.mean([C0_ATE, C1_ATE])



def ATE_matrix(T, Tstar, C, Y):
    """ Ate matrix adjustment derived for our paper. 
        from https://arxiv.org/pdf/1203.3504.pdf
    """
    df = pd.DataFrame({
        'T': list(T),
        'Tstar': list(Tstar),
        'C': list(C),
        'Y': list(Y),
    })

    def eps(c):
        # error type 1: P(T*=0 | T = 1, C)
        subset = df.loc[(df['T'] == 1) & (df['C'] == c)]
        numerator = subset.loc[subset['Tstar'] == 0]
        return float(len(numerator)) / len(subset)

    def delta(c):
        # error type 1: P(T*=1 | T = 0, C)
        subset = df.loc[(df['T'] == 0) & (df['C'] == c)]
        numerator = subset.loc[subset['Tstar'] == 1]
        return float(len(numerator)) / len(subset)

    def joint_Tstar(y, c, tstar):
        # P(Y, C, T*)
        subset = df.loc[(df['Y'] == y) & (df['C'] == c) & (df['Tstar'] == tstar)]
        return float(len(subset)) / len(df)

    def Minverse(c, t, tstar):
        e = eps(c)
        d = delta(c)
        matrix = [
            [1 - e, -e],
            [-d, 1-d]
        ]
        # TODO play with this
        return matrix[t][tstar] / (1 - e - d)

    def joint_T(y, c, t):
        out = 0
        for tsti in set(Tstar):
            out += Minverse(c, t, tsti) * joint_Tstar(y, c, tsti)
        return out

    out = 0
    for ci in set(C):
        pC = float(len(df.loc[df['C'] == ci])) / len(df)
        T1 = joint_T(1, ci, 1) / sum([joint_T(yi, ci, 1) for yi in set(Y)])
        T0 = joint_T(1, ci, 0) / sum([joint_T(yi, ci, 0) for yi in set(Y)])
        tmp = (T1 - T0) * pC
        out += tmp
    return -out


def ATE_me(T, Tstar, C, Y):
    """ ATE matrix adjustment as derived for this paper (eq 3): 
            https://arxiv.org/pdf/1810.00956.pdf
    """
    df = pd.DataFrame({
        'T': list(T),
        'Tstar': list(Tstar),
        'C': list(C),
        'Y': list(Y),
    })

    def eps(c, y):
        # error type 1: P(T=0 | T* = 1, C, Y)
        subset = df.loc[(df['Tstar'] == 1) & (df['C'] == c) & (df['Y'] == y)]
        numerator = subset.loc[subset['T'] == 0]
        return float(len(numerator)) / len(subset)

    def delta(c, y):
        # error type 2: P(T=1 | T* =0, C, Y)
        subset = df.loc[(df['Tstar'] == 0) & (df['C'] == c) & (df['Y'] == y)]
        numerator = subset.loc[subset['T'] == 1]
        return float(len(numerator)) / len(subset)

    def q(c, y, tstar):
        # P(T*=t, C=c, Y=y)
        subset = df.loc[(df['Tstar'] == tstar) & (df['C'] == c) & (df['Y'] == y)]
        return float(len(subset)) / len(df)

    def joint_prob1(c, y):
        # P(T=1, C=c, Y=y)
        num = -delta(c, y) * q(c, y, 0) + (1 - delta(c, y)) * q(c, y, 1)
        denom = 1 - eps(c, y) - delta(c, y)
        return num / denom

    def joint_prob0(c, y):
        # P(T=1, C=c, Y=y)
        num = (1 - eps(c, y)) * q(c, y, 0) - eps(c, y) * q(c, y, 1)
        denom = 1 - eps(c, y) - delta(c, y)
        return num / denom

    def YgivenT1(c, y):
        # P(Y | T=1, C=c)
        num = joint_prob1(c, y)
        denom = sum([joint_prob1(c, yi) for yi in set(Y)])
        return num / denom

    def YgivenT0(c, y):
        # P(Y | T=0, C=c)
        num = joint_prob0(c, y)
        denom = sum([joint_prob0(c, yi) for yi in set(Y)])
        return num / denom

    out = 0
    for ci in set(C):
        pC = float(len(df.loc[df['C'] == ci])) / len(df)
        effect = YgivenT1(ci, 1) - YgivenT0(ci, 1)
        out += effect * pC

    return out


def phi_coef(X1, X2):
    """
        association between feature in text and target
            where target is also binary
        X1/X2 are binary variables
        https://en.wikipedia.org/wiki/Phi_coefficient
    """
    return pearsonr(X1, X2)[0]


def pointwise_biserial(feature, text, targets):
    """ feature: string
        text: list( list(words) )
        targets: list(float)
    pointwise biserial statistic
    https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    """
    s = np.std(targets)

    group0 = []
    group1 = []
    for text_example, val in zip(text, targets):
        if val == -1:
            continue
        if feature in text_example:
            group0.append(val)
        else:
            group1.append(val)

    m0 = np.mean(group0)
    m1 = np.mean(group1)

    n0 = float(len(group0))
    n1 = float(len(group1))
    n = n0 + n1

    rpb = (abs(m1 - m0) / s) * np.sqrt((n0 * n1) / (n ** 2))
    if type(rpb) == type(0.0):
        return None
    return rpb


def cramers_v(feature, text, targets, possible_labels):
    """ feature: string
        text: list( list(words) )
        targets: list(string)
        possible_labels: list(string)
        chisq statistic for a single feature, given some text
        and target info (Y) and possible_labels (possible values for Y)
    """
    obs = np.zeros( (2, len(possible_labels)) )
    for description, target in zip(text, targets):
        if feature in description:
            obs[1, possible_labels.index(target)] += 1
        else:
            obs[0, possible_labels.index(target)] += 1

    row_totals = np.sum(obs, axis=1)
    col_totals = np.sum(obs, axis=0)
    n = np.sum(obs)
    expected = np.outer(row_totals, col_totals) / n
    chisq = np.sum( np.nan_to_num(((obs - expected) ** 2 ) / expected ))

    r = 2
    k = len(possible_labels)
    phisq = chisq / n
    V = np.sqrt(phisq / min(k-1, r-1))
    return V





















