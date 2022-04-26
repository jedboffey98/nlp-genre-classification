import os
from sklearn.pipeline import make_pipeline
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
from sklearn import metrics
from lime.lime_text import LimeTextExplainer
import random
import nltk
from nltk.corpus import stopwords

#from nltk
STOPWORDS = list(stopwords.words('english'))

def compareModelAccuracy(dataFrame, target):
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
    dfTrain, dfTest = train_test_split(dataFrame, test_size=0.15, random_state=12)

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

    for model in models:
        svm = model
        title = model.__class__.__name__
        model = CalibratedClassifierCV(svm, cv=3)

        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
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
        plt.title(title)

        plt.show()

def showAllReports(dataFrame, target):
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
    dfTrain, dfTest = train_test_split(dataFrame, test_size=0.15, random_state=12)

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
        svm = model
        title = model.__class__.__name__
        model = CalibratedClassifierCV(svm, cv=3)

        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
        vectorizer.fit_transform(dfTrain.text)

        pipeline = make_pipeline(vectorizer, model)
        
        pipeline.fit(dfTrain.text, dfTrain[target])
        yPred = pipeline.predict(dfTest.text)
        yPredProb = pipeline.predict_proba(dfTest.text)

        print("\n\n" + title)
        
        accuracy = metrics.accuracy_score(dfTest[target], yPred)
        print("Accuracy:",  round(accuracy,2))
        print("Detail:")
        print(metrics.classification_report(dfTest[target], yPred))

        
        classes = np.unique(dfTest[target])
        y_test_array = pd.get_dummies(dfTest[target], drop_first=False).values

        ## Plot roc
        fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(len(classes)):
            fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
                                yPredProb[:,i])
            ax[0].plot(fpr, tpr, lw=3, 
                    label='{0} (area={1:0.2f})'.format(classes[i], 
                                    metrics.auc(fpr, tpr))
                    )
        ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
        ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
                xlabel='False Positive Rate', 
                ylabel="True Positive Rate (Recall)", 
                title="Receiver operating characteristic")
        ax[0].legend(loc="lower right")
        ax[0].grid(True)
            
        ## Plot precision-recall curve

        for i in range(len(classes)):
            precision, recall, thresholds = metrics.precision_recall_curve(
                        y_test_array[:,i], yPredProb[:,i])
            ax[1].plot(recall, precision, lw=3, 
                    label='{0} (area={1:0.2f})'.format(classes[i], 
                                        metrics.auc(recall, precision))
                    )
        ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
                ylabel="Precision", title="Precision-Recall curve")
        ax[1].legend(loc="best")
        ax[1].grid(True)

        fig.suptitle(title)

        plt.show()

def showLinearSCVMatrices(dataFrame, target):
    dfTrain, dfTest = train_test_split(dataFrame, test_size=0.15, random_state=12)

    #most accurate model
    svm = LinearSVC()
    model = CalibratedClassifierCV(svm, cv=3)

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
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

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
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

def inputtedPrediction(df):
    print("Enter text below:\n")
    text = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        text.append(line)

    text = "\n".join(text)

    dfTrain, _ = train_test_split(df, test_size=0.15, random_state=12)
    genreDf = dfTrain.drop('subgenre', axis=1)
    subgenreDf = dfTrain.drop('genre', axis=1)

    #most accurate model
    svm = LinearSVC()
    model = CalibratedClassifierCV(svm, cv=3)

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, encoding="latin-1", ngram_range=(1, 2), stop_words=STOPWORDS)
    vectorizer.fit_transform(genreDf.text)

    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(genreDf.text, genreDf.genre)
    genreYPred = pipeline.predict([text])
    genreYPredProb = pipeline.predict_proba([text])
    
    print("Pred:", genreYPred[0], "| Prob:", round(np.max(genreYPredProb[0]), 2)) # show explanation
    explainer = LimeTextExplainer(class_names=np.unique(genreDf.genre))
    explained = explainer.explain_instance(text, pipeline.predict_proba, num_features=5)
    print(explained.as_list())

    pipeline.fit(subgenreDf.text, subgenreDf.subgenre)
    subgenreYPred = pipeline.predict([text])
    subgenreYPredProb = pipeline.predict_proba([text])

    print("Pred:", subgenreYPred[0], "| Prob:", round(np.max(subgenreYPredProb[0]), 2)) # show explanation
    explainer = LimeTextExplainer(class_names=np.unique(subgenreDf.subgenre))
    explained = explainer.explain_instance(text, pipeline.predict_proba, num_features=5)
    print(explained.as_list())


if __name__ == "__main__":
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")
    testPath = os.path.join(os.getcwd(), "test-corpus-processed")
    devPath = os.path.join(os.getcwd(), "dev-corpus-processed")

    df = processor.getFrame([trainingPath, testPath, devPath])
    genreDf = df.drop('subgenre', axis=1)
    subgenreDf = df.drop('genre', axis=1)

    nltk.download('stopwords')

    while True:
        print("Menu...")
        print("(1) Show model accuracies for genres")
        print("(2) Show model accuracies for subgenres")
        print("(3) Show SVC genre confusion matrix")
        print("(4) Show SVC subgenre confusion matrix")
        print("(5) Show LimeTextExplainers for genre predictions")
        print("(6) Show LimeTextExplainers for subgenre predictions")
        print("(7) Input text and see prediction, probability, and why")
        print("(8) Show genre score reports for all models")
        print("(9) Show subgenre score reports for all models")
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
        if(inp == "7"):
            inputtedPrediction(df)
        if(inp == "8"):
            showAllReports(genreDf, "genre")
        if(inp == "9"):
            showAllReports(subgenreDf, "subgenre")
