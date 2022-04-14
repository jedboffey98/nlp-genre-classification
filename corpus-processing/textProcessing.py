import os
import re

CORPUS_FOLDERS = ["dev-corpus", "test-corpus", "training-corpus"]

def stripPosTags(path):
    #path is being passed in from a listdir so we shouldn't ever hit this block (possibly redudant?)
    if not os.path.isfile(path):
        print("Error: Missing file (" + path + ")")
        exit() #just close out here and diagnose the problem

    file = open(path)
    text = file.read()

    #processedText = " ".join(word.split("/")[0] for word in text.split())

    processedText = re.sub("\/([^\s]+)", "", text)
    processedText = re.sub('\s+',' ', processedText)
        

    #write to new files
    head, tail = os.path.split(path)
    head = head + "-processed"

    processedPath = os.path.join(head, tail)
    with open(processedPath, "w") as file:
        file.write(processedText)


def main():
    cwd = os.getcwd()
    for folder in CORPUS_FOLDERS:
        path = os.path.join(cwd, folder)
        for file in os.listdir(path):
            print(file)
            stripPosTags(os.path.join(path, file))

if __name__ == "__main__":
    main()