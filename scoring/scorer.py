from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Scorer:
    def __init__(self, answerFilePath):

        file = open(answerFilePath, "r")

        self.catDict = dict()

        lines = file.readlines()
        for line in lines:
            split = line.split()
            self.catDict[split[0]] = split[1]

    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred)
    
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred)

    def scoreAll(self, predictionDict): #parameter should be dict in form key: documentId, value: predictedGenre
        y_pred = []
        y_true = []

        for documentId, predictedGenre in predictionDict.items():
            y_pred.append(predictedGenre)
            y_true.append(self.catDict[documentId])
        
        print("Accuracy score: " + self.accuracy(y_true, y_pred))
        print("Precision score: " + self.precision(y_true, y_pred))
        print("Recall score: " + self.recall(y_true, y_pred))
        print("F1 score: " + self.f1(y_true, y_pred))

    