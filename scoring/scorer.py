from sklearn import metrics
import numpy as np

class Scorer:
    def __init__(self, answerFilePath):

        file = open(answerFilePath, "r")

        self.catDict = dict()

        lines = file.readlines()
        for line in lines:
            split = line.split()
            self.catDict[split[0]] = split[1]

    def accuracy(_, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def precision(_, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average="weighted", labels=np.unique(y_pred))
    
    def recall(_, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average="weighted", labels=np.unique(y_pred))

    def f1(_, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average="weighted", labels=np.unique(y_pred))

    def confusionMatrix(_, y_true, y_pred):
        return metrics.classification_report(y_true, y_pred, labels=np.unique(y_pred))

    def scoreAll(self, predictionDict): #parameter should be dict in form key: documentId, value: predictedGenre
        y_pred = []
        y_true = []

        for documentId, predictedGenre in predictionDict.items():
            y_pred.append(predictedGenre)
            y_true.append(self.catDict[documentId])

        print("Accuracy score: " + str(self.accuracy(y_true, y_pred)))
        print("Precision score: " + str(self.precision(y_true, y_pred)))
        print("Recall score: " + str(self.recall(y_true, y_pred)))
        print("F1 score: " + str(self.f1(y_true, y_pred)))

        print("\n\n")

        print(self.confusionMatrix(y_true, y_pred))

    