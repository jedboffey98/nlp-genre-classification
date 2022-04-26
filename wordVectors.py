import gensim
import tfidfProcessor as processor
import os
from sklearn.model_selection import train_test_split
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt
from keras import preprocessing as kprocessing
import seaborn as sns

def getNGrams():
    trainingPath = os.path.join(os.getcwd(), "training-corpus-processed")
    testPath = os.path.join(os.getcwd(), "test-corpus-processed")
    devPath = os.path.join(os.getcwd(), "dev-corpus-processed")

    df = processor.getFrame([trainingPath, testPath, devPath])

    dfTrain, _ = train_test_split(df, test_size=0.15, random_state=12)
    corpus = dfTrain.text

    ## create list of lists of unigrams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i+1]) 
                    for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)

    ## detect bigrams and trigrams
    """
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, 
                    delimiter=" ".encode(), min_count=5, threshold=10)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], 
                delimiter=" ".encode(), min_count=5, threshold=10)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
    """

    ## fit w2v
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=300,   
    window=8, min_count=1, sg=1, epochs=30)

    word = "hypothesis"
    fig = plt.figure()## word embedding
    tot_words = [word] + [tupla[0] for tupla in 
                    nlp.wv.most_similar(word, topn=20)]
    X = nlp.wv[tot_words]## pca to reduce dimensionality from 300 to 3
    pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
    X = pca.fit_transform(X)## create dtf
    dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
    dtf_["input"] = 0
    dtf_["input"].iloc[0:1] = 1## plot 3d
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dtf_[dtf_["input"]==0]['x'], 
            dtf_[dtf_["input"]==0]['y'], 
            dtf_[dtf_["input"]==0]['z'], c="black")
    ax.scatter(dtf_[dtf_["input"]==1]['x'], 
            dtf_[dtf_["input"]==1]['y'], 
            dtf_[dtf_["input"]==1]['z'], c="red")
    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], 
        yticklabels=[], zticklabels=[])
    for label, row in dtf_[["x","y","z"]].iterrows():
        x, y, z = row
        ax.text(x, y, z, s=label)
    
    plt.show()

    ## tokenize text
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', 
                        oov_token="NaN", 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index## create sequence
    lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)## padding sequence
    X_train = kprocessing.sequence.pad_sequences(lst_text2seq, 
                        maxlen=15, padding="post", truncating="post")

    sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False)
    plt.show()


if __name__ == "__main__":
    getNGrams()