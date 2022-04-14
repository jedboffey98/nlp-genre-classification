import os
from sklearn.pipeline import make_pipeline
import tfidfProcessor as processor
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from lime.lime_text import LimeTextExplainer
import random

def main():
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")
    testPath = os.path.join(os.getcwd(), "test-corpus-processed")
    devPath = os.path.join(os.getcwd(), "dev-corpus-processed")

    #fetch a frame of all the documents, we'll split the data later with sklearn
    df = processor.getFrame([trainingPath, testPath, devPath])

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
        KNeighborsClassifier(n_neighbors=3, metric="cosine")
    ]

    CV = 3
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding="latin-1", ngram_range=(1, 2), stop_words="english")

    dfTrain, dfTest = train_test_split(df, test_size=0.25, random_state=12)

    features = tfidf.fit_transform(dfTrain.text).toarray()
    testFeatures = tfidf.transform(dfTest.text).toarray()

    labels = dfTrain.genre
    for model in models:
        modelName = model.__class__.__name__

        accuracies = cross_val_score(model, features, labels, scoring="accuracy", cv=CV)
        for foldIdx, accuracy in enumerate(accuracies):
            entries.append((modelName, foldIdx, accuracy))

    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])

    #plot accuracies on bar chart
    sns.boxplot(x="model_name", y="accuracy", data=cv_df)
    sns.stripplot(x="model_name", y="accuracy", data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)

    plt.show()

    print(cv_df.groupby("model_name").accuracy.mean())

    #most accurate model
    svm = LinearSVC()
    model = CalibratedClassifierCV(svm)
    
    model.fit(features, dfTrain.genre)
    yPred = model.predict(testFeatures)
    yPredProb = model.predict_proba(testFeatures)
    
    conf_mat = confusion_matrix(dfTest.genre, yPred)
    
    #plot confusion matrix
    plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=np.unique(df.genre.values), yticklabels=np.unique(df.genre.values))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.show()

    i = random.randint(0, len(dfTest) - 1)
    textExample = dfTest.iloc[i].text
    #textExample = Xtest[i]

    c = make_pipeline(tfidf, model)

    # check true value and predicted value
    print("True:", dfTest.iloc[i].genre, "--> Pred:", yPred[i], "| Prob:", round(np.max(yPredProb[i]), 2)) # show explanation
    explainer = LimeTextExplainer(class_names=np.unique(dfTest.genre))
    explained = explainer.explain_instance(textExample, c.predict_proba, num_features=5)
    print(explained.as_list())

if __name__ == "__main__":
    main()