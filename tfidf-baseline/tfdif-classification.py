from cmath import cos
from fileinput import filename
import os
import math

#Stopwords from homework assignment
STOPWORDS = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

CORPUS_FOLDERS = ["dev-corpus", "test-corpus", "training-corpus"]

def vectorize(s):
    text = s.replace("\n", " ")
    text = "".join(c for c in text if c.isalpha() or c == " ")
    text = text.strip(os.linesep)

    text = text.split(" ")

    text = [word for word in text if word not in STOPWORDS]
    text = [word for word in text if word != ""]

    textDict = dict.fromkeys(text, 0)

    for word in text:
        textDict[word] += 1 / len(text)

    return textDict

def cosineSimilarity(d1, d2):
    numerator = 0
    denominator = 0

    for k1, v1, in d1.items():
        numerator += v1 * d2.get(k1, 0)
        denominator += v1 * v1

    denominator2 = 0
    for v2 in d2.values():
        denominator2 += v2 * v2
    
    return numerator / math.sqrt(denominator * denominator2)

def getCats():
    filePath = os.path.join(os.getcwd(), "corpus-info\\cats.txt")
    file = open(filePath, "r")

    catDict = dict()

    lines = file.readlines()
    for line in lines:
        split = line.split()
        catDict[split[0]] = split[1]
    
    print(catDict)
    return catDict

def main():
    trainingDocuments = dict()
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")

    for fileName in os.listdir(trainingPath):
        file = open(os.path.join(trainingPath, fileName), "r")
        data = file.read()
        tokenizedDocument = vectorize(data)

        trainingDocuments[fileName] = tokenizedDocument

    cats = getCats()
    
    devPath = os.path.join(os.getcwd(), "dev-corpus-processed")

    correct = 0
    incorrect = 0

    for fileName in os.listdir(devPath):
        file = open(os.path.join(devPath, fileName), "r")
        data = file.read()

        testDict = vectorize(data)
        cosSims = dict()

        for fID, trainingDict in trainingDocuments.items():
            sim = cosineSimilarity(testDict, trainingDict)

            cosSims[fID] = sim
        
        #for fID, cosSim in dict(sorted(cosSims.items(), key=lambda item: item[1], reverse=True)).items():
            #print("Similarity score of " + str(fileName) + " to " + str(fID) + " of: " + str(cosSim))

        closest = max(cosSims, key=cosSims.get)

        print("Most similar to " + closest + "(" + cats[closest] + ")")
        if cats[closest] == cats[fileName]:
            print("Correct")
            correct += 1
        else:
            print("Incorrect")
            incorrect += 1

    print("Correct: " + str(correct) + ". Incorrect: " + str(incorrect) + ". Percentage correct: " + str(correct / (correct + incorrect)) + ".")
    

if __name__ == "__main__":
    main()