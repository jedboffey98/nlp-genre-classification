import os

def getCats():
    filePath = os.path.join(os.getcwd(), "corpus-info\\cats.txt")
    file = open(filePath, "r")

    catDict = dict()

    lines = file.readlines()
    for line in lines:
        split = line.split()
        catDict[split[0]] = split[1]
    
    return catDict