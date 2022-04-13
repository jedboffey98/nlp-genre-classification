import os
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
from sklearn.metrics import confusion_matrix

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

    features = tfidf.fit_transform(df.text).toarray()
    labels = df.genre
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
    model = LinearSVC()
    Xtrain, Xtest, yTrain, yTest, _, _ = train_test_split(features, labels, df.index, test_size=0.25, random_state=0)
    model.fit(Xtrain, yTrain)
    yPred = model.predict(Xtest)
    
    conf_mat = confusion_matrix(yTest, yPred)
    
    #plot confusion matrix
    plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=np.unique(df.genre.values), yticklabels=np.unique(df.genre.values))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.show()

if __name__ == "__main__":
    main()