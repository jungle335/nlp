import json
import re
import enchant
import matplotlib.pyplot as plt
import numpy as np
import spacy
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.stem.snowball import SnowballStemmer
from num2words import num2words
# spacy.cli.download("ro_core_news_sm")
from spacy.lang import ro

nlp = spacy.load("ro_core_news_sm")
stemmer = SnowballStemmer(language='romanian')


def notSmallChars(content):
    chars, pattern = set(), re.compile("[^a-z]")
    for match in pattern.finditer(content):
        chars.update(match.group())
    return chars


def translateNumber(content):
    pattern = re.compile(r'[\d]+')
    for match in pattern.finditer(content):
        content = content.replace(match.group(), num2words(match.group(), lang='ro'))
    return content


def plotting(freq, labels, nrOfWords=25):
    plt.figure(figsize=(15, 7))
    plt.title(labels[0])
    plt.xticks(rotation=50)
    plt.ylabel(labels[1])
    # cele mai frecvente 25 cuvinte
    plt.bar([elem[0] for elem in freq][:nrOfWords], [elem[1] for elem in freq][:nrOfWords], orientation='vertical')
    plt.show()


def read(numeFisier):
    allChars, levenstein, words, frE, frG, tokens = set(), [], [], {}, {}, {}

    with open(numeFisier) as f:
        reviews = json.load(f)['reviews']
        for review in reviews:
            content = review['content']
            allChars.update(notSmallChars(content))
            content = translateNumber(content)
            content = re.sub(r"(www.|https?:)[\S]+", "", content)  # remove links
            content = re.sub(r"[.?!%:;,\-()\"]+", "", content)  # remove punctuation marks
            content = re.sub(r"\s+", " ", content)  # remove useless spaces
            content = [token.text for token in nlp(content)]  # tokenize
            words.extend(content)  # all tokens in one place

            if len(content) not in tokens:
                tokens[len(content)] = 1
            else:
                tokens[len(content)] += 1

            values, frequencies = np.unique(content, return_counts=True)  # frequencies tokens
            for i in range(len(values)):
                if values[i] not in frE:
                    frE[values[i]] = frequencies[i]
                else:
                    frE[values[i]] += frequencies[i]

            withoutStopwords = [text for text in content if text not in ro.STOP_WORDS]

            # stemming, lemming
            stem = [stemmer.stem(word) for word in withoutStopwords]
            values, frequencies = np.unique(stem, return_counts=True)  # frequencies stemming
            for i in range(len(values)):
                if values[i] not in frG:
                    frG[values[i]] = frequencies[i]
                else:
                    frG[values[i]] += frequencies[i]

            content = " ".join(withoutStopwords)  # remake string after removing stopwords
            lemma = [word.lemma_ for word in nlp(content)]
            # stem != lema
            for i in range(len(withoutStopwords)):
                levenstein.append((withoutStopwords[i], enchant.utils.levenshtein(stem[i], lemma[i])))

    # caractere
    print("\nCaractere diferite de litere mici")
    print(allChars)

    # 15 cuvinte top
    levenstein = sorted(list(set(levenstein)), reverse=True, key=lambda t: t[1])
    print("\nPrimele 15 cuvinte top")
    print([elem[0] for elem in levenstein[:15]])

    # top 20 trigrame
    trigramMeas = TrigramAssocMeasures()
    collFounder = TrigramCollocationFinder.from_words(words)
    trigramRes = collFounder.nbest(trigramMeas.pmi, 20)
    print("\nTop 20 trigrame")
    print(trigramRes)

    # plot 2
    wordFreq = [(val, frE[val]) for val in frE]
    wordFreq.sort(reverse=True, key=lambda t: t[1])
    plotting(wordFreq, ["", "Frequency"])
    wordFreq = [(val, frG[val]) for val in frG]
    wordFreq.sort(reverse=True, key=lambda t: t[1])
    plotting(wordFreq, ["", "Frequency"])

    # la primul plot se observa o frecventa mai mare a tokenurilor intrucat sunt incluse stopwords, iar la urmatorul
    # mai putine intrucat operatia de stemming este aplicata dupa eliminarea stopwords

    # plot 3
    plt.figure(figsize=(5, 5))
    plt.xlabel("Number of tokens")
    plt.ylabel("Number of reviews")
    plt.bar(tokens.keys(), tokens.values(), orientation='vertical')
    plt.show()

    # se observa o compactare a numarului de reviewuri la cele pozitive spre deosebire de
    # cele negative


read('positive_reviews.json')
read('negative_reviews.json')