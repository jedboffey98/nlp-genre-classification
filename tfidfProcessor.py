import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pandas as pd
from utils import getCats
from utils import getKey
import numpy as np

CORPUS_FOLDERS = ["dev-corpus", "test-corpus", "training-corpus"]

def vectorize(s):
    #clean text
    text = s.replace("\n", " ")
    text = "".join(c for c in text if c.isalpha() or c == " ")
    text = text.strip(os.linesep)
    text = text.split(" ")
    text = text.lower()
    text = [word for word in text if word != ""]

    vector = dict.fromkeys(text, 0)

    #convert to tf-idf
    for word in text:
        vector[word] += 1 / len(text)

    return vector

def getFrame(filePaths):
    completeDf = pd.DataFrame(columns=["documentId", "text"])

    key = getKey()

    for path in filePaths:
        for fileName in os.listdir(path):
            file = open(os.path.join(path, fileName), "r")
            data = file.read()
            data = data.lower()

            df = pd.DataFrame([[fileName, data]], columns=["documentId", "text"])
            
            completeDf = pd.concat([completeDf, df])
    

    completeDf = pd.concat([completeDf.set_index("documentId"), key.set_index("documentId")], axis=1)

    return completeDf

def getFeatures(trainingPath):
    trainingDf = getFrame(trainingPath)    

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding="latin-1", ngram_range=(1, 2), stop_words="english")

    features = tfidf.fit_transform(trainingDf.text).toarray()
    labels = trainingDf.genre

    printMostCorrelated(tfidf, features, labels, getCats())

    return features

def printMostCorrelated(tfidf, features, labels, cats):
    uniqueGenres = np.unique(list(cats.values()))

    for genre in uniqueGenres:
        features_chi2 = chi2(features, labels == genre)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

        print("# '{}':".format(genre))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-2:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))

def getManualVectors(trainingPath, testPath):
    trainingVectors = dict()
    testVectors = dict()

    for fileName in os.listdir(trainingPath):
        file = open(os.path.join(trainingPath, fileName), "r")
        data = file.read()
        vectorized = vectorize(data)

        trainingVectors[fileName] = vectorized

    for fileName in os.listdir(testPath):
        file = open(os.path.join(testPath, fileName), "r")
        data = file.read()
        vectorized = vectorize(data)

        testVectors[fileName] = vectorized
    
    return trainingVectors, testVectors

if __name__ == "__main__":
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")
    testPath = os.path.join(os.getcwd(), "dev-corpus-processed")
    print(testPath)
    getFrame(testPath)