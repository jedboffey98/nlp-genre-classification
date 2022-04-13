import os

CORPUS_FOLDERS = ["dev-corpus", "test-corpus", "training-corpus"]

def vectorize(s):
    #clean text
    text = s.replace("\n", " ")
    text = "".join(c for c in text if c.isalpha() or c == " ")
    text = text.strip(os.linesep)
    text = text.split(" ")
    text = [word for word in text if word != ""]

    vector = dict.fromkeys(text, 0)

    #convert to tf-idf
    for word in text:
        vector[word] += 1 / len(text)

    return vector

def getVectors(trainingPath, testPath):
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