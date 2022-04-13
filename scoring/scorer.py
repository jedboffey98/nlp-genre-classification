class Scorer:
    def __init__(self, answerFilePath):

        file = open(answerFilePath, "r")

        self.catDict = dict()

        lines = file.readlines()
        for line in lines:
            split = line.split()
            self.catDict[split[0]] = split[1]

    def isCorrect(self, documentId, predictedGenre):
        return predictedGenre  == self.catDict[documentId]

    def scoreAll(self, predictionDict): #parameter should be dict in form key: documentId, value: predictedGenre
        for documentId, predictedGenre in predictionDict.items():
            correct = self.isCorrect(documentId, predictedGenre)

    