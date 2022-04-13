import math
import os
import tfidfProcessor as processor
from utils import getCats
from scoring import scorer

def cosineSimilarity(d1, d2):
    numerator = 0
    denominator = 0

    for k1, v1, in d1.items():
        numerator += v1 * d2.get(k1, 0)
        denominator += v1 * v1

    denominator2 = 0
    for v2 in d2.values():
        denominator2 += v2 * v2

    d = math.sqrt(denominator * denominator2)

    if d != 0:
        return numerator / d

    return 0

def main():
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")
    testPath = os.path.join(os.getcwd(), "dev-corpus-processed")

    trainingVectors, testVectors = processor.getVectors(trainingPath, testPath)

    preds = dict()
    answerCats = getCats()

    for testId, testVector in testVectors.items():
        similarities = dict()

        for trainId, trainingVector in trainingVectors.items():
            sim = cosineSimilarity(testVector, trainingVector)
            similarities[trainId] = sim

        closestTrainId = max(similarities, key=similarities.get)
        preds[testId] = answerCats[closestTrainId]
    
    sc = scorer.Scorer(os.path.join(os.getcwd(), "corpus-info\\cats.txt"))
    sc.scoreAll(preds)

if __name__ == "__main__":
    main()