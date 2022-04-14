import os
import pandas as pd

def getCats():
    filePath = os.path.join(os.getcwd(), "corpus-info\\cats.txt")
    file = open(filePath, "r")

    catDict = dict()

    lines = file.readlines()
    for line in lines:
        split = line.split()
        catDict[split[0]] = split[2]
    
    return catDict

def getKey():
    filePath = os.path.join(os.getcwd(), "corpus-info\\cats.txt")

    df = pd.read_csv(filePath, sep=" ", header=None, names=["documentId", "genre", "subgenre"])

    return df

if __name__ == "__main__":
    getKey()