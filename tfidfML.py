import os
from sklearn import pipeline
from sklearn.pipeline import Pipeline, make_pipeline
import tfidfProcessor as processor
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from lime.lime_text import LimeTextExplainer
import random

def compareModelAccuracy(dataFrame, target):
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words="english")
    dfTrain, _ = train_test_split(dataFrame, test_size=0.15, random_state=12)

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
        KNeighborsClassifier(n_neighbors=3, metric="cosine")
    ]

    entries = []
    cv = KFold(n_splits=3)

    for model in models:
        modelName = model.__class__.__name__

        pipeline = make_pipeline(vectorizer, model)

        accuracies = cross_val_score(pipeline, dfTrain.text, dfTrain[target], scoring="accuracy", cv=cv, error_score="raise")

        for foldIdx, accuracy in enumerate(accuracies):
            entries.append((modelName, foldIdx, accuracy))
        
    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])

    #plot accuracies on bar chart
    sns.boxplot(x="model_name", y="accuracy", data=cv_df)
    sns.stripplot(x="model_name", y="accuracy", data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)

    plt.show()

def showLinearSCVMatrices(dataFrame, target):
    dfTrain, dfTest = train_test_split(dataFrame, test_size=0.15, random_state=12)

    #most accurate model
    svm = LinearSVC()
    model = CalibratedClassifierCV(svm, cv=3)

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words="english")
    vectorizer.fit_transform(dfTrain.text)

    pipeline = make_pipeline(vectorizer, model)
    
    pipeline.fit(dfTrain.text, dfTrain[target])
    yPred = pipeline.predict(dfTest.text)
    
    conf_mat = confusion_matrix(dfTest[target], yPred)
    
    #plot confusion matrix
    plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=np.unique(df[target].values), yticklabels=np.unique(df[target].values), cmap="summer")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.show()

def showRandomLimeExplainer(dataFrame, target):
    dfTrain, dfTest = train_test_split(dataFrame, test_size=0.15, random_state=12)

    #most accurate model
    svm = LinearSVC()
    model = CalibratedClassifierCV(svm, cv=3)

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words="english")
    vectorizer.fit_transform(dfTrain.text)

    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(dfTrain.text, dfTrain[target])
    yPred = pipeline.predict(dfTest.text)
    yPredProb = pipeline.predict_proba(dfTest.text)

    inp = ""
    while inp != "exit":
        i = random.randint(0, len(dfTest) - 1)

        textExample = dfTest.iloc[i].text

        print("\nTrue:", dfTest.iloc[i][target], "--> Pred:", yPred[i], "| Prob:", round(np.max(yPredProb[i]), 2)) # show explanation
        explainer = LimeTextExplainer(class_names=np.unique(dfTest[target]))
        explained = explainer.explain_instance(textExample, pipeline.predict_proba, num_features=5)
        print(explained.as_list())

        inp = input("Type [exit] to return to the menu or hit enter to see more explainers\n")


if __name__ == "__main__":
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")
    testPath = os.path.join(os.getcwd(), "test-corpus-processed")
    devPath = os.path.join(os.getcwd(), "dev-corpus-processed")

    df = processor.getFrame([trainingPath, testPath, devPath])
    genreDf = df.drop('subgenre', axis=1)
    subgenreDf = df.drop('genre', axis=1)

    while True:
        print("Menu...")
        print("(1) Show model accuracies for genres")
        print("(2) Show model accuracies for subgenres")
        print("(3) Show SVC genre confusion matrix")
        print("(4) Show SVC subgenre confusion matrix")
        print("(5) Show LimeTextExplainers for genre predictions")
        print("(6) Show LimeTextExplainers for subgenre predictions")
        print("(Exit)")

        inp = input("")
        
        if(inp.lower() == "exit"):
            break

        if(inp == "1"):
            compareModelAccuracy(genreDf, "genre")
        if(inp == "2"):
            compareModelAccuracy(subgenreDf, "subgenre")
        if(inp == "3"):
            showLinearSCVMatrices(genreDf, "genre")
        if(inp == "4"):
            showLinearSCVMatrices(subgenreDf, "subgenre")
        if(inp == "5"):
            showRandomLimeExplainer(genreDf, "genre")
        if(inp == "6"):
            showRandomLimeExplainer(subgenreDf, "subgenre")