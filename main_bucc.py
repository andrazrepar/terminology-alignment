# coding=utf-8

import pandas as pd
from nltk import word_tokenize
from collections import defaultdict
import editdistance
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn import pipeline
import argparse
import csv
import pickle

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from sklearn import svm
from transliterate import translit, get_available_language_codes


def longest_common_substring(s):
    s = s.split('\t')
    s1, s2 = s[0], s[1]
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def longest_common_subsequence(s):
    s = s.split('\t')
    a, b = s[0], s[1]
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result

def isWordCognate(s, idx):
    terms = s.split('\t')
    term_source, term_target = terms[0], terms[1]
    word_source, word_target = term_source.split()[idx], term_target.split()[idx]
    word_pair = word_source + '\t' + word_target
    lcs = longest_common_substring(word_pair)
    lgth = max(len(word_source), len(word_target))
    if lgth > 3 and float(len(lcs))/lgth >= 0.7:
        #print(s, lcs, lgth)
        return 1
    return 0



def isFirstWordTranslated(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    firstWordSource = term1.split()[0].strip()
    firstWordTarget = term2.split()[0].strip()
    for target, p in giza_dict[firstWordSource]:
        if target == firstWordTarget:
            return 1

    #fix for compounding problem
    if len(term2) > 4:
        for target, p in giza_dict[firstWordSource]:
            if term2.startswith(target):
                #print(term2, target)
                return 1
    return 0


def isLastWordTranslated(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    lastWordSource = term1.split()[-1].strip()
    lastWordTarget = term2.split()[-1].strip()
    for target, p in giza_dict[lastWordSource]:
        if target == lastWordTarget:
            return 1

    # fix for compounding problem
    if len(term2) > 4:
        for target, p in giza_dict[lastWordSource]:
            if term2.endswith(target):
                return 1
    return 0



def percentageOfTranslatedWords(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter+=1
                break
    return float(counter)/len(term1)


def percentageOfNotTranslatedWords(s, giza_dict):
    return 1 - percentageOfTranslatedWords(s, giza_dict)


def longestTranslatedUnitInPercentage(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter += 1
                if counter > max:
                    max = counter
                break
        else:
            counter = 0
    return float(max) / len(term1)


def longestNotTranslatedUnitInPercentage(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter = 0
                break
        else:
            counter += 1
            if counter > max:
                max = counter
    return float(max) / len(term1)


def wordLengthMatch(x):
    terms = x.split('\t')
    term_source, term_target = terms[0], terms[1]
    if len(term_source.split()) == len(term_target.split()):
        return 1
    return 0


def sourceTermLength(x):
    terms = x.split('\t')
    term_source, _ = terms[0], terms[1]
    return len(term_source.split())


def targetTermLength(x):
    terms = x.split('\t')
    _, term_target = terms[0], terms[1]
    return len(term_target.split())


def isWordCovered(x, giza_dict, index):
    terms = x.split('\t')
    term_source, term_target = terms[0], terms[1]
    for word, score in giza_dict[term_source.split()[index]]:
        if word in term_target.split():
            return 1
    lcstr = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / max(len(term_source.split()[index]), len(term_target))
    lcsr = float(len(longest_common_subsequence(term_source.split()[index] + '\t' + term_target))) / max(len(term_source.split()[index]), len(term_target))
    dice = 2 * float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / (len(term_source.split()[index]) + len(term_target))
    nwd = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / min(len(term_source.split()[index]),len(term_target))
    editDistance = 1 - (float(editdistance.eval(term_source.split()[index], term_target)) / max(len(term_source.split()[index]), len(term_target)))
    if max(lcstr, lcsr, nwd, dice, editDistance) > 0.7:
        return 1
    return 0



def isWordCoveredEmbeddings(x, embedding_dict, index):
    terms = x.split('\t')
    term_source, term_target = terms[0], terms[1]
    try:
        for word in embedding_dict[term_source.split()[index]]:
            if word == term_target.split():
                return 1
    except:
        score = 0


    lcstr = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / max(
        len(term_source.split()[index]), len(term_target))
    lcsr = float(len(longest_common_subsequence(term_source.split()[index] + '\t' + term_target))) / max(
        len(term_source.split()[index]), len(term_target))
    dice = 2 * float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / (
    len(term_source.split()[index]) + len(term_target))
    nwd = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / min(
        len(term_source.split()[index]), len(term_target))
    editDistance = 1 - (float(editdistance.eval(term_source.split()[index], term_target)) / max(
        len(term_source.split()[index]), len(term_target)))
    if max(lcstr, lcsr, nwd, dice, editDistance) > 0.7:
        return 1
    return 0


def percentageOfCoverage(x, giza_dict):
    terms = x.split('\t')
    length = len(terms[0].split())
    counter = 0
    for index in range(length):
        counter += isWordCovered(x, giza_dict, index)
    return counter/length

def percentageOfCoverageEmbeddings(x, embedding_dict):
    terms = x.split('\t')
    length = len(terms[0].split())
    counter = 0
    for index in range(length):
        counter += isWordCoveredEmbeddings(x, embedding_dict, index)
    return counter/length

### embeddings features

def isFirstWordMatch(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    firstWordSource = term1.split()[0].strip()
    firstWordTarget = term2.split()[0].strip()
    #print(term1, term2)
    #print(firstWordSource, firstWordTarget)
    #print(embedding_dict[firstWordSource])
    try:
        if embedding_dict[firstWordSource][0] == firstWordTarget:
            return 1
        
        #fix for compounding problem
        if len(term2) > 4:
            target = embedding_dict[firstWordSource][0]
            if term2.startswith(target):
                #print(term2, target)
                return 1
        return 0
    except:
        return 0

def isLastWordMatch(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    firstWordSource = term1.split()[-1].strip()
    firstWordTarget = term2.split()[-1].strip()

    try:
        if embedding_dict[firstWordSource][0] == firstWordTarget:
            return 1
        
        #fix for compounding problem
        if len(term2) > 4:
            target = embedding_dict[firstWordSource][0]
            if term2.startswith(target):
                #print(term2, target)
                return 1
        return 0
    except:
        return 0

def percentageOfFirstMatchWords(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    for word in term1:
        try:
            target = embedding_dict[word][0]
        except:
            target = 'xyz123'  ### bedno ampak 
        if target in term2:
            counter+=1
    return float(counter)/len(term1)

def percentageOfNotFirstMatchWords(s, embedding_dict):
    return 1 - percentageOfFirstMatchWords(s, embedding_dict)

def longestFirstMatchUnitInPercentage(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        try:
            target = embedding_dict[word]
        except:
            target = 'xyz123'
        if target in term2:
            counter += 1
            if counter > max:
                max = counter
            break
        else:
            counter = 0
    return float(max) / len(term1)


def longestNotFirstMatchUnitInPercentage(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        try:
            target = embedding_dict[word]
        except:
            target = 'xyz123'
        if target in term2:
            counter = 0
            break
        else:
            counter += 1
            if counter > max:
                max = counter
    return float(max) / len(term1)

def isFirstWordTopnMatch(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    firstWordSource = term1.split()[0].strip()
    firstWordTarget = term2.split()[0].strip()
    try:
        for target in embedding_dict[firstWordSource]:
            if target == firstWordTarget:
                return 1

        #fix for compounding problem
        if len(term2) > 4:
            for target in embedding_dict[firstWordSource]:
                if term2.startswith(target):
                    #print(term2, target)
                    return 1
        return 0
    except:
        return 0


def isLastWordTopnMatch(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    lastWordSource = term1.split()[-1].strip()
    lastWordTarget = term2.split()[-1].strip()
    try:
        for target in embedding_dict[lastWordSource]:
            if target == lastWordTarget:
                return 1

        # fix for compounding problem
        if len(term2) > 4:
            for target in embedding_dict[lastWordSource]:
                if term2.endswith(target):
                    return 1
        return 0
    except:
        return 0
    



def percentageOfTopnMatchWords(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    try:
        for word in term1:
            for target in embedding_dict[word]:
                if target in term2:
                    counter+=1
                    break
        return float(counter)/len(term1)
    except:
        return 0


def percentageOfNotTopnMatchWords(s, embedding_dict):
    return 1 - percentageOfTopnMatchWords(s, embedding_dict)


def longestTopnMatchUnitInPercentage(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    try:
        for word in term1:
            for target in embedding_dict[word]:
                if target in term2:
                    counter += 1
                    if counter > max:
                        max = counter
                    break
            else:
                counter = 0
        return float(max) / len(term1)
    except:
        return 0


def longestNotTopnMatchUnitInPercentage(s, embedding_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    try:
        for word in term1:
            for target in embedding_dict[word]:
                if target in term2:
                    counter = 0
                    break
            else:
                counter += 1
                if counter > max:
                    max = counter
        return float(max) / len(term1)
    except:
        return 0


def preprocess(text):
    tokens = word_tokenize(text)
    return " ".join(tokens).lower()


def transcribe(text, lang):
    sl_repl = {'č':'ch', 'š':'sh', 'ž': 'zh'}
    en_repl = {'x':'ks', 'y':'j', 'w':'v', 'q':'k'}
    fr_repl = {'é':'e', 'à':'a', 'è':'e', 'ù':'u', 'â':'a', 'ê':'e', 'î':'i', 'ô':'o', 'û':'u', 'ç':'c', 'ë':'e', 'ï':'i', 'ü':'u'}
    nl_repl = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ï':'i', 'ü': 'u', 'ë':'e', 'ö':'o', 'à':'a', 'è':'e', 'ĳ':'ij'}
    if lang == 'en':
        en_tr = [en_repl.get(item,item) for item in list(text)]
        return "".join(en_tr).lower()
    elif lang == 'sl':
        sl_tr = [sl_repl.get(item,item)  for item in list(text)]
        return "".join(sl_tr).lower()
    elif lang == 'fr':
        fr_tr = [fr_repl.get(item, item) for item in list(text)]
        return "".join(fr_tr).lower()
    elif lang == 'nl':
        nl_tr = [nl_repl.get(item, item) for item in list(text)]
        return "".join(nl_tr).lower()
    elif lang == 'et':
        et_tr = translit(text, 'ru')
        return(et_tr)



def arrangeData(input):
    dd = defaultdict(list)
    with open(input, encoding='utf8') as f:
        for line in f:
            try:
                source, target, score = line.split()
                source = source.strip('`’“„,‘')
                target = target.strip('`’“„,‘')
                dd[source].append((target, score))
            except:
                pass
                #print(line)

    for k, v in dd.items():
        v = sorted(v, key=lambda tup: float(tup[1]), reverse=True)
        new_v = []
        for word, p in v:
            if (len(k) < 4 and len(word) > 5) or (len(word) < 4 and len(k) > 5):
                continue
            if float(p) < 0.05:
                continue
            new_v.append((word, p))
        dd[k] = new_v
    return dd


def arrangeDistances(input, topn):
    i = 0
    distances = {}
    with open(input, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        for line in r:
            source = line[0].lower()
            target = line[1:]
            if source not in distances:
                distances[source] = [target]
            else:
                distances[source].append(target)
            #i = i + 1
            #if i % 100000 == 0:
            #    print('Lines read:', i)

    print('loading done')
    #pickle.dump(distances, open('eurovoc_sl-en-distances_fasttext.p', 'wb'))
    
    #distances = pickle.load(open('fasttext/eurovoc_sl-en-distances_fasttext.p', 'rb'))
    topn_distances = {}
    topn_distances_with_values = {}
    j = 0
    for source, target in distances.items():
        sorted_target = sorted(target, key=lambda x: float(x[1]), reverse=True)[:int(topn)]
        new_target = [t[0].lower() for t in sorted_target]
        new_target2 = [[t[0].lower(), t[1]] for t in sorted_target]
        topn_distances[source] = new_target
        topn_distances_with_values[source] = new_target2
        #if j % 100 == 0:
        #    print('words done:', j)
        #j = j + 1

    #print(len(topn_distances))
    pickle.dump(topn_distances, open(input+'.p', 'wb'))
    
    #topn_distances = pickle.load(open('fasttext/eurovoc_sl-en-top2_distances_fasttext.p', 'rb'))
    #print(len(topn_distances))
    return topn_distances, topn_distances_with_values

        #for source, target in distances.items():
        #    w.writerow([source,] + flat_list(sorted(target, key=lambda x: float(x[1]), reverse=True)[:3]))

def arrangeDistances_boshko(input, source_words_file, target_words_file, topn):
    source_words = []
    with open(source_words_file, "r") as f:
        for line in f:
            source_words.append(line.strip())
    
    target_words = []
    with open(target_words_file, "r") as f:
        for line in f:
            target_words.append(line.strip())

    i = 0
    distances = {}
    with open(input, 'r') as f:
        r = csv.reader(f, delimiter=',')
        for line in r:
            source = source_words[int(line[0])]
            target = target_words[int(line[1])]
            score = float(line[2])
            if source not in distances:
                distances[source] = [[target, score]]
            else:
                distances[source].append([target, score])
            #i = i + 1
            #if i % 100000 == 0:
            #    print('Lines read:', i)

    print('loading done')
    #pickle.dump(distances, open('eurovoc_sl-en-distances_fasttext.p', 'wb'))
    
    #distances = pickle.load(open('fasttext/eurovoc_sl-en-distances_fasttext.p', 'rb'))
    topn_distances = {}
    topn_distances_with_values = {}
    j = 0
    for source, target in distances.items():
        sorted_target = sorted(target, key=lambda x: float(x[1]), reverse=True)[:int(topn)]
        new_target = [t[0].lower() for t in sorted_target]
        new_target2 = [[t[0].lower(), t[1]] for t in sorted_target]
        topn_distances[source] = new_target
        topn_distances_with_values[source] = new_target2
        #print(source, new_target)
        #if j % 100 == 0:
        #    print('words done:', j)
        #j = j + 1

    #print(len(topn_distances))
    
    #topn_distances = pickle.load(open('fasttext/eurovoc_sl-en-top2_distances_fasttext.p', 'rb'))
    #print(len(topn_distances))
    return topn_distances, topn_distances_with_values


def createExamples(df_term_pairs, neg_train_count):
    term_pairs = list(zip(df_term_pairs['SRC'].tolist(), df_term_pairs['TAR'].tolist()))
    tar_terms = [x[1] for x in term_pairs]
    src_terms = [x[0] for x in term_pairs]

 

    #make sure train and test don't overlap
    existing_pairs = set()

    #build train set
    train_set = []
    for src_term, tar_term in term_pairs:
        counter = neg_train_count
        existing_pairs.add(src_term + '\t' + tar_term)
        train_set.append((src_term, tar_term, 1))
        while counter > 0:
            neg_example = random.choice(tar_terms)
            while (neg_example == tar_term):
                neg_example = random.choice(tar_terms)
            train_set.append((src_term, neg_example, 0))
            existing_pairs.add(src_term + '\t' + neg_example)
            counter -= 1
    df_train = pd.DataFrame(train_set, columns=['src_term', 'tar_term', 'label'])

    return df_train

def createTestSet(source_terms_file, target_terms_file):
    with open(source_terms_file, "r") as file:
        next(file)
        src_terms = [line.strip() for line in file]
    
    with open(target_terms_file, "r") as file:
        next(file)
        tar_terms = [line.strip() for line in file]

    pairs = []
    for src_term in src_terms:
        for tar_term in tar_terms:
            pairs.append((src_term, tar_term))

    return pd.DataFrame(pairs, columns=['src_term', 'tar_term'])


def filterTrainSet(df, ratio, fasttext_topn, cognates=False, onlyFasttext=False):
    print("Train set shape pre filter: ", df.shape)
    df_pos = df[df['label'] == 1]
    print(fasttext_topn)

    df_pos_dict = df_pos[df_pos['isFirstWordTranslated'] == 1]
    df_pos_dict = df_pos_dict[df_pos['isLastWordTranslated'] == 1]
    df_pos_dict = df_pos_dict[df_pos['isFirstWordTranslated_reversed'] == 1]
    df_pos_dict = df_pos_dict[df_pos['isLastWordTranslated_reversed'] == 1]
    df_pos_dict = df_pos_dict[df_pos['percentageOfCoverage'] > 0.66]
    df_pos_dict = df_pos_dict[df_pos['percentageOfCoverage_reversed'] > 0.66]

    df_pos_dict.reset_index(drop=True, inplace=True)

    df_pos_fasttext_1 = df_pos[df_pos['isFirstWordMatch'] == 1]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['isLastWordMatch'] == 1]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['isFirstWordMatch_reversed'] == 1]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['isLastWordMatch_reversed'] == 1]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['percentageOfCoverageEmbeddings'] > 0.66]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['percentageOfCoverageEmbeddings_reversed'] > 0.66]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['percentageOfFirstMatchWords'] > 0.66]
    df_pos_fasttext_1 = df_pos_fasttext_1[df_pos['percentageOfFirstMatchWords_reversed'] > 0.66]
    df_pos_fasttext_n = df_pos[df_pos['isFirstWordTopnMatch'] == 1]
    df_pos_fasttext_n = df_pos_fasttext_n[df_pos['isLastWordTopnMatch'] == 1]
    df_pos_fasttext_n = df_pos_fasttext_n[df_pos['isFirstWordTopnMatch_reversed'] == 1]
    df_pos_fasttext_n = df_pos_fasttext_n[df_pos['isLastWordTopnMatch_reversed'] == 1]
    df_pos_fasttext_n = df_pos_fasttext_n[df_pos['percentageOfTopnMatchWords'] > 0.66]
    df_pos_fasttext_n = df_pos_fasttext_n[df_pos['percentageOfTopnMatchWords_reversed'] > 0.66]


    df_pos_fasttext_1.reset_index(drop=True, inplace=True)
    #df_pos_fasttext_n.reset_index(drop=True, inplace=True)

    if cognates:
        df_pos_cognate_1 = df_pos[df_pos['isFirstWordMatch'] == 1]
        df_pos_cognate_1 = df_pos_cognate_1[df_pos['isLastWordCognate'] == 1]

        df_pos_cognate_2 = df_pos[df_pos['isLastWordMatch'] == 1]
        df_pos_cognate_2 = df_pos_cognate_2[df_pos['isFirstWordCognate'] == 1]

        df_pos_cognate_3 = df_pos[df_pos['isFirstWordCognate'] == 1]
        df_pos_cognate_3 = df_pos_cognate_3[df_pos['isLastWordCognate'] == 1]

        df_pos_cognate_1.reset_index(drop=True, inplace=True)
        df_pos_cognate_2.reset_index(drop=True, inplace=True)
        df_pos_cognate_3.reset_index(drop=True, inplace=True)

        #df_pos = pd.concat([df_pos_fasttext_1, df_pos_cognate_1, df_pos_cognate_2, df_pos_cognate_3])
        #df_pos = pd.concat([df_pos_fasttext_1, df_pos_fasttext_n, df_pos_cognate_1, df_pos_cognate_2, df_pos_cognate_3])
        df_pos = pd.concat([df_pos_dict, df_pos_fasttext_1, df_pos_fasttext_n, df_pos_cognate_1, df_pos_cognate_2, df_pos_cognate_3])
        df_pos = df_pos.drop_duplicates()
    else:
        df_pos = pd.concat([df_pos_dict, df_pos_fasttext_1, df_pos_fasttext_n])
        #df_pos = pd.concat([df_pos_dict, df_pos_fasttext_1])
        #df_pos = df_pos_fasttext_1
        df_pos = df_pos.drop_duplicates()
    


    df_neg = df[df['label'] == 0].sample(frac=1, random_state=123)[:df_pos.shape[0] * ratio]
    df_neg.reset_index(drop=True, inplace=True)

    df = pd.concat([df_pos, df_neg])
    print("Train set shape post filter: ", df.shape)
    return df

def createEmbeddingFeatures(data, featureType, distances):
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    print('preprocessing done')
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']
    data['isFirstWordMatch_' + featureType] = data['term_pair'].map(lambda x: isFirstWordMatch(x, distances))
    data['isLastWordMatch_' + featureType] = data['term_pair'].map(lambda x: isLastWordMatch(x, distances))
    data['percentageOfFirstMatchWords_' + featureType] = data['term_pair'].map(lambda x: percentageOfFirstMatchWords(x, distances))
    data['percentageOfNotFirstMatchWords_' + featureType] = data['term_pair'].map(lambda x: percentageOfNotFirstMatchWords(x, distances))
    data['longestFirstMatchUnitInPercentage_' + featureType] = data['term_pair'].map(lambda x: longestFirstMatchUnitInPercentage(x, distances))
    data['longestNotFirstMatchUnitInPercentage_' + featureType] = data['term_pair'].map(lambda x: longestNotFirstMatchUnitInPercentage(x, distances))

    data['isFirstWordTopnMatch_' + featureType] = data['term_pair'].map(lambda x: isFirstWordTopnMatch(x, distances))
    data['isLastWordTopnMatch_' + featureType] = data['term_pair'].map(lambda x: isLastWordTopnMatch(x, distances))
    data['percentageOfTopnMatchWords_' + featureType] = data['term_pair'].map(lambda x: percentageOfTopnMatchWords(x, distances))
    data['percentageOfNotTopnMatchWords_' + featureType] = data['term_pair'].map(lambda x: percentageOfNotTopnMatchWords(x, distances))
    data['longestTopnMatchUnitInPercentage_' + featureType] = data['term_pair'].map(lambda x: longestTopnMatchUnitInPercentage(x, distances))
    data['longestNotTopnMatchUnitInPercentage_' + featureType] = data['term_pair'].map(lambda x: longestNotTopnMatchUnitInPercentage(x, distances))

    data['isFirstWordCoveredEmbeddings_' + featureType] = data['term_pair'].map(lambda x: isWordCoveredEmbeddings(x, distances, 0))
    data['isLastWordCoveredEmbeddings_' + featureType] = data['term_pair'].map(lambda x: isWordCoveredEmbeddings(x, distances, -1))
    data['percentageOfCoverageEmbeddings_' + featureType] = data['term_pair'].map(lambda x: percentageOfCoverageEmbeddings(x, distances))
    data['percentageOfNonCoverageEmbeddings_' + featureType] = data['term_pair'].map(lambda x: 1 - percentageOfCoverageEmbeddings(x, distances))
    data['diffBetweenCoverageAndNonCoverageEmbeddings_' + featureType] = data['percentageOfCoverageEmbeddings_' + featureType] - data['percentageOfNonCoverageEmbeddings_' + featureType]

    data = data.drop(['term_pair'], axis = 1)

    return data



def createFeatures(data, giza_dict, giza_dict_reversed, distances, distances_reversed, fasttext_topn, cognates=False):
    data['src_term'] = data['src_term'].map(lambda x: preprocess(x))
    data['tar_term'] = data['tar_term'].map(lambda x: preprocess(x))
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    print('preprocessing done')

    data['isFirstWordTranslated'] = data['term_pair'].map(lambda x: isFirstWordTranslated(x, giza_dict))
    data['isLastWordTranslated'] = data['term_pair'].map(lambda x: isLastWordTranslated(x, giza_dict))
    data['percentageOfTranslatedWords'] = data['term_pair'].map(lambda x: percentageOfTranslatedWords(x, giza_dict))
    data['percentageOfNotTranslatedWords'] = data['term_pair'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict))
    data['longestTranslatedUnitInPercentage'] = data['term_pair'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict))
    data['longestNotTranslatedUnitInPercentage'] = data['term_pair'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict))

    data['term_pair'] = data['tar_term'] + '\t' + data['src_term']

    data['isFirstWordTranslated_reversed'] = data['term_pair'].map(lambda x: isFirstWordTranslated(x, giza_dict_reversed))
    data['isLastWordTranslated_reversed'] = data['term_pair'].map(lambda x: isLastWordTranslated(x, giza_dict_reversed))
    data['percentageOfTranslatedWords_reversed'] = data['term_pair'].map(lambda x: percentageOfTranslatedWords(x, giza_dict_reversed))
    data['percentageOfNotTranslatedWords_reversed'] = data['term_pair'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict_reversed))
    data['longestTranslatedUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict_reversed))
    data['longestNotTranslatedUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict_reversed))
    
    data['src_term_tr'] = data['src_term'].map(lambda x: transcribe(x, 'en'))
    data['tar_term_tr'] = data['tar_term'].map(lambda x: transcribe(x, 'sl'))
    data['term_pair_tr'] = data['src_term_tr'] + '\t' + data['tar_term_tr']
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']
    #print(data['term_pair_tr'])

    if cognates:
        data['isFirstWordCognate'] = data['term_pair_tr'].map(lambda x: isWordCognate(x, 0))
        data['isLastWordCognate'] = data['term_pair_tr'].map(lambda x: isWordCognate(x, -1))

    data['longestCommonSubstringRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['longestCommonSubsequenceRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_subsequence(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['dice'] = data['term_pair_tr'].map(lambda x: (2 * float(len(longest_common_substring(x)))) / (len(x.split('\t')[0]) + len(x.split('\t')[1])))
    data['NWD'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / min(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['editDistanceNormalized'] = data['term_pair_tr'].map(lambda x: 1 - (float(editdistance.eval(x.split('\t')[0], x.split('\t')[1])) / max(len(x.split('\t')[0]), len(x.split('\t')[1]))))

    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    data['isFirstWordCovered'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict, 0))
    data['isLastWordCovered'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict, -1))
    data['percentageOfCoverage'] = data['term_pair'].map(lambda x: percentageOfCoverage(x, giza_dict))
    data['percentageOfNonCoverage'] = data['term_pair'].map(lambda x: 1 -percentageOfCoverage(x, giza_dict))
    data['diffBetweenCoverageAndNonCoverage'] = data['percentageOfCoverage'] - data['percentageOfNonCoverage']

    #if cognates:
    data['wordLengthMatch'] = data['term_pair'].map(lambda x: wordLengthMatch(x))
    data['sourceTermLength'] = data['term_pair'].map(lambda x: sourceTermLength(x))
    data['targetTermLength'] = data['term_pair'].map(lambda x: targetTermLength(x))

    data['term_pair'] = data['tar_term'] + '\t' + data['src_term']

    data['isFirstWordCovered_reversed'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict_reversed, 0))
    data['isLastWordCovered_reversed'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict_reversed, -1))
    data['percentageOfCoverage_reversed'] = data['term_pair'].map(lambda x: percentageOfCoverage(x, giza_dict_reversed))
    data['percentageOfNonCoverage_reversed'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict_reversed))
    data['diffBetweenCoverageAndNonCoverage_reversed'] = data['percentageOfCoverage_reversed'] - data['percentageOfNonCoverage_reversed']
    data['averagePercentageOfTranslatedWords'] = (data['percentageOfTranslatedWords'] + data['percentageOfTranslatedWords_reversed']) / 2

    if fasttext_topn > 0:
        data['term_pair'] = data['src_term'] + '\t' + data['tar_term']
        
        data['isFirstWordMatch'] = data['term_pair'].map(lambda x: isFirstWordMatch(x, distances))
        data['isLastWordMatch'] = data['term_pair'].map(lambda x: isLastWordMatch(x, distances))
        data['percentageOfFirstMatchWords'] = data['term_pair'].map(lambda x: percentageOfFirstMatchWords(x, distances))
        data['percentageOfNotFirstMatchWords'] = data['term_pair'].map(lambda x: percentageOfNotFirstMatchWords(x, distances))
        data['longestFirstMatchUnitInPercentage'] = data['term_pair'].map(lambda x: longestFirstMatchUnitInPercentage(x, distances))
        data['longestNotFirstMatchUnitInPercentage'] = data['term_pair'].map(lambda x: longestNotFirstMatchUnitInPercentage(x, distances))

        data['isFirstWordTopnMatch'] = data['term_pair'].map(lambda x: isFirstWordTopnMatch(x, distances))
        data['isLastWordTopnMatch'] = data['term_pair'].map(lambda x: isLastWordTopnMatch(x, distances))
        data['percentageOfTopnMatchWords'] = data['term_pair'].map(lambda x: percentageOfTopnMatchWords(x, distances))
        data['percentageOfNotTopnMatchWords'] = data['term_pair'].map(lambda x: percentageOfNotTopnMatchWords(x, distances))
        data['longestTopnMatchUnitInPercentage'] = data['term_pair'].map(lambda x: longestTopnMatchUnitInPercentage(x, distances))
        data['longestNotTopnMatchUnitInPercentage'] = data['term_pair'].map(lambda x: longestNotTopnMatchUnitInPercentage(x, distances))

        data['isFirstWordCoveredEmbeddings'] = data['term_pair'].map(lambda x: isWordCoveredEmbeddings(x, distances, 0))
        data['isLastWordCoveredEmbeddings'] = data['term_pair'].map(lambda x: isWordCoveredEmbeddings(x, distances, -1))
        data['percentageOfCoverageEmbeddings'] = data['term_pair'].map(lambda x: percentageOfCoverageEmbeddings(x, distances))
        data['percentageOfNonCoverageEmbeddings'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverageEmbeddings(x, distances))
        data['diffBetweenCoverageAndNonCoverageEmbeddings'] = data['percentageOfCoverageEmbeddings'] - data['percentageOfNonCoverageEmbeddings']

        data['term_pair'] = data['tar_term'] + '\t' + data['src_term']
        
        data['isFirstWordMatch_reversed'] = data['term_pair'].map(lambda x: isFirstWordMatch(x, distances_reversed))
        data['isLastWordMatch_reversed'] = data['term_pair'].map(lambda x: isLastWordMatch(x, distances_reversed))
        data['percentageOfFirstMatchWords_reversed'] = data['term_pair'].map(lambda x: percentageOfFirstMatchWords(x, distances_reversed))
        data['percentageOfNotFirstMatchWords_reversed'] = data['term_pair'].map(lambda x: percentageOfNotFirstMatchWords(x, distances_reversed))
        data['longestFirstMatchUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestFirstMatchUnitInPercentage(x, distances_reversed))
        data['longestNotFirstMatchUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestNotFirstMatchUnitInPercentage(x, distances_reversed))

        data['isFirstWordTopnMatch_reversed'] = data['term_pair'].map(lambda x: isFirstWordTopnMatch(x, distances_reversed))
        data['isLastWordTopnMatch_reversed'] = data['term_pair'].map(lambda x: isLastWordTopnMatch(x, distances_reversed))
        data['percentageOfTopnMatchWords_reversed'] = data['term_pair'].map(lambda x: percentageOfTopnMatchWords(x, distances_reversed))
        data['percentageOfNotTopnMatchWords_reversed'] = data['term_pair'].map(lambda x: percentageOfNotTopnMatchWords(x, distances_reversed))
        data['longestTopnMatchUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestTopnMatchUnitInPercentage(x, distances_reversed))
        data['longestNotTopnMatchUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestNotTopnMatchUnitInPercentage(x, distances_reversed))

        data['isFirstWordCoveredEmbeddings_reversed'] = data['term_pair'].map(lambda x: isWordCoveredEmbeddings(x, distances_reversed, 0))
        data['isLastWordCoveredEmbeddings_reversed'] = data['term_pair'].map(lambda x: isWordCoveredEmbeddings(x, distances_reversed, -1))
        data['percentageOfCoverageEmbeddings_reversed'] = data['term_pair'].map(lambda x: percentageOfCoverageEmbeddings(x, distances_reversed))
        data['percentageOfNonCoverageEmbeddings_reversed'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverageEmbeddings(x, distances_reversed))
        data['diffBetweenCoverageAndNonCoverageEmbeddings_reversed'] = data['percentageOfCoverageEmbeddings_reversed'] - data['percentageOfNonCoverageEmbeddings_reversed']
        

    data = data.drop(['term_pair', 'term_pair_tr', 'src_term_tr', 'tar_term_tr'], axis = 1)


    print('feature construction done')
    return data


def build_manual_eval_set(terms_src, terms_tar, iter_src, iter_tar):
    # one iteration == 100 source terms * 100 target terms
    all_terms = []
    #print(terms_src, terms_tar)
    i = 0
    for src_term in terms_src[(iter_src-1) * 100:iter_src * 100]:
        j = 0
        for tar_term in terms_tar[(iter_tar-1) * 100:iter_tar * 100]:
            #print(i, j)
            j = j + 1
            all_terms.append([str(src_term).strip(), str(tar_term).strip()])
        i = i + 1
    df = pd.DataFrame(all_terms)
    #print(all_terms)
    print(df.shape)
    df.columns = ['src_term', 'tar_term']
    
    return df

def filterByWordLength(df):
    df_pos = df[df['prediction'] == 1]
    df_neg = df[df['prediction'] == 0]
    df_neg.reset_index(drop=True, inplace=True)
    df_pos.reset_index(drop=True, inplace=True)
    for i, row in df_pos.iterrows():
        if len(row['src_term'].split()) != len(row['tar_term'].split()):
            #df_pos.set_value(i, 'prediction', 0) apparently this was removed in new pandas version: https://stackoverflow.com/questions/60294463/attributeerror-dataframe-object-has-no-attribute-set-value
            df_pos.at[i, 'prediction'] = 0
    df = pd.concat([df_pos, df_neg])
    return df


class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(['src_term', 'tar_term'], axis=1)
        return hd_searches.values


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train and test a machine translation model')
    parser.add_argument('--pretrained_dataset', type=str, default='length',
                        help='Use one of the already generated train and test sets from the data set folder. '
                             'Used for reproduction of results reported in the paper. Argument options are: aker, giza_terms_only, clean and unbalanced')
    parser.add_argument('--trainset_balance', type=str, default='1',
                        help='Define the ratio between positive and negative examples in trainset, e.g. 200 means that 200 negative examples are generated'
                             'for every positive example in the train set')
    parser.add_argument('--testset_balance', type=str, default='200',
                        help='Define the ratio between positive and negative examples in testset, e.g. 200 means that 200 negative examples are generated'
                             'for every positive example in the initial term list.')
    parser.add_argument('--giza_only', default='False', help='Use only terms that are found in the giza corpus')
    parser.add_argument('--giza_clean', default='False', help='Use clean version of Giza++ generated dictionary')
    parser.add_argument('--filter_trainset', default='False', help='Filter train set')
    parser.add_argument('--cognates', default='False', help='Approach that improves recall for cognate terms')
    parser.add_argument('--term_length_filter', default='False', help='Additional filter which removes all positively classified terms whose word length do not match')
    parser.add_argument('--predict_source', default='', help='Use your model on your own dataset. Value should be a path to a list of source language terms')
    parser.add_argument('--predict_target', default='', help='Use your model on your own dataset. Value should be a path to a list of target language terms')
    parser.add_argument('--fasttext_topn', type=int, default=0, help='Size of topn array for fasttext embeddings features. 0 means no fastext features are generated.')
    parser.add_argument('--exclusive_fasttext', default='False', help='use only fasttext and cognate features.')
    parser.add_argument('--lang', default='sl', help='Possible values are sl, fr and nl for Slovene, French and Dutch target language')
    parser.add_argument('--runname', default="runname", help='custom filename')

    start_time = time.time()

    params = parser.parse_args()


    #assert params.pretrained_dataset in ['', 'aker', 'giza_terms_only', 'clean', 'unbalanced', 'cognates', 'fasttext', 'unbalanced-new'], 'Allowed arguments are: aker, giza_terms_only, clean, unbalanced, cognates'

    pretrained_dataset = params.pretrained_dataset
    trainset_balance = int(params.trainset_balance)
    testset_balance = int(params.testset_balance)
    giza_only = True if params.giza_only=='True' else False
    filter = True if params.filter_trainset=='True' else False
    cognates = True if params.cognates == 'True' else False
    giza_clean = True if params.giza_clean == 'True' else False
    predict_source = params.predict_source
    predict_target = params.predict_target
    term_length_filter = True if params.term_length_filter == 'True' else False
    fasttext_topn = int(params.fasttext_topn)
    exclusive_fasttext = True if params.exclusive_fasttext=='True' else False
    lang = params.lang
    run_name = params.runname
    

    print("Lemmatized approach, arguments: ")
    print("Pretrained dataset: ", pretrained_dataset)
    print("Trainset balance: ", trainset_balance)
    print("Testset balance: ", testset_balance)
    print("Giza terms only: ", giza_only)
    print("Filter trainset features: ", filter)
    print("Cognates: ", cognates)
    print("Term length filter: ", term_length_filter)
    print("Giza clean: ", giza_clean)
    print("Fasttext TopN: ", fasttext_topn)
    print("Only Fasttext: ", exclusive_fasttext)
    print("Run name: ", run_name)

    folder = "bucc2022_training/"

    ## bucc dataset
    bucc_fasttext_location = "fasttext/bucc_mw_en-fr-distances_fasttext.tsv"
    bucc_fasttext_reversed_location = "fasttext/bucc_mw_fr-en-distances_fasttext.tsv"

    bucc_en_words = "fasttext/bucc_words/bucc_en_words.csv"
    bucc_fr_words = "fasttext/bucc_words/bucc_fr_words.csv"
    bucc_boshko_embeddings = {
        "distilbert": "fasttext/bucc_words/bucc_distilbert-base-nli-mean-tokens.csv",
        "MiniLM": "fasttext/bucc_words/bucc_all-MiniLM-L6-v2.csv",
        "mpnet": "fasttext/bucc_words/bucc_all-mpnet-base-v2.csv",
        "roberta": "fasttext/bucc_words/bucc_roberta-large-nli-stsb-mean-tokens.csv",
        "xlm": "fasttext/bucc_words/bucc_xlm-r-large-en-ko-nli-ststb.csv"
    }

    ## še embeddingi na osnovi evrovoca

    eurovoc_en_words = "fasttext/eurovoc_words/eurovoc_en_words.csv"
    eurovoc_fr_words = "fasttext/eurovoc_words/eurovoc_fr_words.csv"
    eurovoc_boshko_embeddings = {
        "distilbert_eurovoc": "fasttext/eurovoc_words/distilbert-base-nli-mean-tokens.csv",
        "MiniLM_eurovoc": "fasttext/eurovoc_words/eurovoc_all-MiniLM-L6-v2.csv",
        "mpnet_eurovoc": "fasttext/eurovoc_words/eurovoc_all-mpnet-base-v2.csv",
        "roberta_eurovoc": "fasttext/eurovoc_words/roberta-large-nli-stsb-mean-tokens.csv",
        "xlm_eurovoc": "fasttext/eurovoc_words/xlm-r-large-en-ko-nli-ststb.csv"
    }

    if giza_clean:
        dd = arrangeData('not_lemmatized/en-' + lang + 'TransliterationBased.txt')
        dd_reversed = arrangeData('not_lemmatized/' + lang + '-enTransliterationBased.txt')
    else:
        dd = arrangeData('not_lemmatized/en-' + lang + 'Unfiltered.txt')
        dd_reversed = arrangeData('not_lemmatized/' + lang + '-enUnfiltered.txt')

    print("arrange giza done.")
    distances, distancesWithValues = arrangeDistances(bucc_fasttext_location, fasttext_topn)
    distances_reversed, distancesWithValues_reversed = arrangeDistances(bucc_fasttext_reversed_location, fasttext_topn)
    print("arrange fasttext distances done.")

    df_terms = pd.read_csv('term_list_' + lang + '.csv', sep=';')
    data_train = createExamples(df_terms, neg_train_count=trainset_balance)
    print("create examples done.")
    data_train = createFeatures(data_train, dd, dd_reversed, distances, distances_reversed, fasttext_topn, cognates=cognates)
    print("create giza, cognate and fasttext features done.")

    arrangedDistances = {}
    for be in bucc_boshko_embeddings:
        distances_bucc, distancesWithValues_bucc = arrangeDistances_boshko(bucc_boshko_embeddings[be], bucc_en_words, bucc_fr_words, fasttext_topn)
        arrangedDistances[be] = distances_bucc
        print("arrange bucc distances done,", be)
        data_train = createEmbeddingFeatures(data_train, be, distances_bucc)
        print("create bucc features done,", be)

    for be in eurovoc_boshko_embeddings:
        distances_eurovoc, distancesWithValues_eurovoc = arrangeDistances_boshko(eurovoc_boshko_embeddings[be], eurovoc_en_words, eurovoc_fr_words, fasttext_topn)
        arrangedDistances[be] = distances_eurovoc
        print("arrange eurovoc distances done,", be)
        data_train = createEmbeddingFeatures(data_train, be, distances_eurovoc)
        print("create eurovoc features done,", be)
    

    print("Train set size: ", data_train.shape[0])
    if filter:
        data_train = filterTrainSet(data_train, trainset_balance, fasttext_topn, cognates=cognates, onlyFasttext=exclusive_fasttext)
    
    data_train.to_csv(folder + "results/training_examples-" + run_name + ".csv", encoding="utf8", index=False, sep='\t')
    #data_train.to_csv('bucc2022_test_enfr_nogold/results/bucc_training_examples-200.csv', encoding="utf8", index=False, sep='\t')

    cols = data_train.columns.values.tolist()
    print('Number of features: ', len(cols) - 3)

    
    # build classification model
    y = data_train['label'].values
    X = data_train.drop(['label'], axis=1)
    svm = svm.SVC(C=10, probability=True)


    features = [('cst', digit_col())]

    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=features,
            n_jobs=8
        )),
        ('scale', Normalizer()),
        ('svm', svm)])

    clf.fit(X, y)
    #pickle.dump(clf, open("bucc2022_test_enfr_nogold/results/bucc_model-200.p", 'wb'))
    pickle.dump(clf, open(folder + "results/model-" + run_name + ".csv", 'wb'))
    print("creating model done.")

    
    # create test set
    #data_test = createTestSet("bucc2022_training/terms-en-small.txt", "bucc2022_training/terms-fr-small.txt")
    data_test = createTestSet(folder + "terms-en.txt", folder + "terms-fr.txt")
    print("create test examples done.")
    data_test = createFeatures(data_test, dd, dd_reversed, distances, distances_reversed, fasttext_topn, 
    cognates=cognates)
    print("create test features done.")
    for be in bucc_boshko_embeddings:
        data_test = createEmbeddingFeatures(data_test, be, arrangedDistances[be])
        print("create bucc test features done,", be)

    for be in eurovoc_boshko_embeddings:
        data_test = createEmbeddingFeatures(data_test, be, arrangedDistances[be])
        print("create eurovoc test features done,", be)

    
    data_test.to_csv(folder + "/results/bucc_test_examples-" + run_name + ".csv", encoding="utf8", index=False, sep='\t')
    y_pred = clf.predict_proba(data_test)

    result = pd.concat([data_test, pd.DataFrame(y_pred, columns=['not_pair', "pair"])], axis=1)

    if term_length_filter:
        result = filterByWordLength(result)

    result.to_csv(folder + "results/all-" + run_name + ".csv", encoding="utf8", index=False)
    result = result.loc[result['pair'] > 0.5]
    result = result[['src_term', 'tar_term', 'pair']]
    result.to_csv(folder + "results/above50-" + run_name + ".csv", encoding="utf8", index=False)

    print(time.time() - start_time, "seconds")























