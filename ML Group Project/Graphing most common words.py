#yesdocs array of strings
#nodocs


import nltk
import collections
import json
import pandas as pd
from collections import Counter
from numpy.core.fromnumeric import size
import pandas as pd
import string
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')


def ratingToNum(ratingString):
    if(ratingString==('no factual content')):
        return +1
    elif(ratingString==('mostly false')):
        return -1
    elif(ratingString==('mixture of true and false')):
        return -1
    elif(ratingString==('mostly true')):
        return +1
    else:
        print("OTHER RATING: ", ratingString)


def prep(review):
    # Remove non-letters
    #review = re.sub("[^a-zA-Z]", " ", review)
    review = re.sub("[0-9]", " ", review)


    review = re.sub("www", " ", review)
    
    # Lower case
    review = review.lower()
    
    # Tokenize to each word.
    token = nltk.word_tokenize(review, language='english')
    
    lemmatizer = nltk.WordNetLemmatizer()
    # Stemming
    #review = [nltk.stem.SnowballStemmer('english').stem(w) for w in token]
    review = [lemmatizer.lemmatize(w) for w in token]
    
    # Join the words back into one string separated by space, and return the result.
    return " ".join(review)


def main():
    #f = open('MoreComments.json')
    #f = open('AllCommentswB.json')
    f = open('finalComments.json')

    articles = json.load(f)
    all_docs = []
    for article in articles:
        art = []
        all_comments = ''
        art.append(ratingToNum(article['TruthRating']))
        for comment in article['Comments']:
            all_comments += prep(comment)
        art.append(all_comments)
        all_docs.append(art)

    df = pd.DataFrame(all_docs, columns=['rating', 'comments'])
    df = df.sort_values(by='rating')
    print(df.head())

    def get_token_list(doc):
        t = doc.lower()
        for p in string.punctuation:
            t = t.replace(p, ' ')
        return t.split()

    #alldocs = nodocs + yesdocs

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words='english', min_df=0.1)

    tovectorize = [a[1] for a in all_docs]

    #print(tovectorize.dtype)
    #for vect in tovectorize:
    #    print(vect)

    tfidf = vectorizer.fit_transform(tovectorize)

    tfidf = tfidf.toarray()

    words = vectorizer.get_feature_names()

    #print(words[:10])

    import matplotlib.pyplot as plt

    #plt.imshow(tfidf)
    #plt.show()
    import numpy as np

    x = tfidf
    y = df['rating']
    y = np.array(y)
    print(x.shape)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score



    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print("Classification Report for Baseline Model")

    fdist = nltk.FreqDist(y_train)
    most_com_pred = fdist.max()
    yBLpred = [most_com_pred] * len(y_test)
    print(classification_report(y_test, yBLpred))

    print("Confusion Matrix for Baseline Model")
    print(confusion_matrix(y_test, yBLpred))
    print(f"Precision: {precision_score(y_test, yBLpred)}")
    print(f"Recall: {recall_score(y_test, yBLpred)}")
    print(f"F1 Score: {f1_score(y_test, yBLpred)}")
    print(f"Accuracy: {accuracy_score(y_test, yBLpred)}")

    linearl1String = "Linear SVM (L1)"
    linearl2String = "Linear SVM (L2)"
    logisticString = "Logistic Regression"



    for modelType in [linearl1String, linearl2String, logisticString]:
        train_mean_error=[]
        test_mean_error=[]
        train_std_error=[]
        test_std_error=[]

        f1array = []
        Cvalues = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        for Cval in Cvalues:
            if(modelType == linearl1String):
                model = LinearSVC(penalty='l1', dual=False, C=Cval)
            elif(modelType == linearl2String):
                model = LinearSVC(penalty='l2', C=Cval)
            else:
                model = LogisticRegression(penalty='l2', C=Cval)
            temp_train = []
            temp_test = []
            scores = cross_val_score(model, x, y, cv=5, scoring='f1')
            train_mean_error.append(np.array(scores).mean())
            train_std_error.append(np.array(scores).std())


        plt.style.use("bmh")
        plt.errorbar(Cvalues, train_mean_error, yerr=train_std_error)
        plt.title(f'{modelType} Cross Validation with regards to C Value')

        plt.xlabel("C Values")
        plt.ylabel("F1 Score")
        plt.xscale('log')

        plt.show()

        bestCvalIndex = np.argmax(train_mean_error)

        bestCval = Cvalues[bestCvalIndex]

        if(modelType == linearl1String):
            model = LinearSVC(penalty='l1', dual=False, C=bestCval).fit(x_train, y_train)
        elif(modelType == linearl2String):
            model = LinearSVC(penalty='l2', C=bestCval).fit(x_train, y_train)
        else:
            model = LogisticRegression(penalty='l2', C=bestCval).fit(x_train, y_train)
        
        y_pred = model.predict(x_test)
        
        print(f"Classification report for {modelType} at C={bestCval}")
        print(classification_report(y_test, y_pred)) 

        print(f"Confusion Matrix for {modelType} at C={bestCval}")
        print(confusion_matrix(y_test, y_pred))
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


        #y_predict = [int(p[1] > 0.5) for p in model.predict_proba(x_test)]
        print(model.coef_.shape)

        coef = model.coef_.reshape(-1)
        print(coef.shape)
        np.argmax(coef)
        # #plt.plot(words, coef[3], label='Mostly True')
        # //plt.plot(words, coef, label='Mostly True')
        # #plt.plot(words, coef[0], label='Mostly False')
        # //plt.legend()
        # //plt.xticks(words, words, rotation='vertical')
        # plt.show()
        idx = np.argsort(abs(coef))[-20:]
        print(idx)

        words = np.array(words)

        newdf = pd.DataFrame({'words': words, 'coefs': list(coef)}, columns=['words', 'coefs'])
        dfsorted = newdf.sort_values('coefs')
        plt.style.use("bmh")
        
        plt.bar('words', 'coefs', data=dfsorted, width=1.0, color=(dfsorted['coefs']>0).map({True:'g',
                                                                        False:'r'}))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.grid(False)
        plt.axhline(linewidth=1, color='0.4')
        plt.title(f"All word weights for {modelType}")

        plt.show()

        newdfT20 = pd.DataFrame({'words': words[idx], 'coefs': list(coef[idx])}, columns=['words', 'coefs'])
        dfsortedT20 = newdfT20.sort_values('coefs')
        plt.style.use("bmh")

        plt.bar('words', 'coefs', data=dfsortedT20, color=(dfsortedT20['coefs']>0).map({True:'g',
                                                                        False:'r'}))
        plt.xticks(words[idx], words[idx], rotation='vertical')
        plt.title(f"Top 20 word weights for {modelType}")
        plt.grid(True)
        plt.show()

    # train_mean_error=[]
    # test_mean_error=[]
    # train_std_error=[]
    # test_std_error=[]

    # f1array = []
    # Cvalues = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # gammas = [1, 4, 16, 64, 256]
    # for Cval in Cvalues:
    #     for gamma in gammas:
    #         model = SVC(C=Cval, gamma=gamma)
    #         temp_train = []
    #         temp_test = []
    #         scores = cross_val_score(model, x, y, cv=5, scoring='f1')
    #         train_mean_error.append(np.array(scores).mean())
    #         train_std_error.append(np.array(scores).std())

    # plt.errorbar(Cvalues, train_mean_error, yerr=train_std_error)
    # plt.title("SVC Cross Validation with regards to C Value")

    # plt.xlabel("C Values")
    # plt.ylabel("F1 Score")
    # plt.xscale('log')
    # plt.show()

    # bestCvalIndex = np.argmax(train_mean_error)

    # bestCval = Cvalues[bestCvalIndex]

    # if(modelType == linearString):
    #     model = LinearSVC(penalty='l2', C=bestCval).fit(x_train, y_train)
    # elif(modelType == KRidgeString):
    #     model = KernelRidge(alpha=(1/(2*bestCval))).fit(x_train, y_train)


    # else:
    #     model = LogisticRegression(penalty='l2', C=bestCval).fit(x_train, y_train)
    
    # y_pred = model.predict(x_test)
    
    # print(f"Classification report for {modelType} at C={bestCval}")
    # print(classification_report(y_test, y_pred)) 


    # #y_predict = [int(p[1] > 0.5) for p in model.predict_proba(x_test)]
    # print(model.coef_.shape)

    # coef = model.coef_.reshape(-1)
    # print(coef.shape)
    # np.argmax(coef)
    # #plt.plot(words, coef[3], label='Mostly True')
    # plt.plot(words, coef, label='Mostly True')
    # #plt.plot(words, coef[0], label='Mostly False')
    # plt.legend()
    # plt.xticks(words, words, rotation='vertical')
    # plt.show()
    # idx = np.argsort(abs(coef))[-20:]
    # print(idx)

    # words = np.array(words)
    # newdf = pd.DataFrame({'words': words[idx], 'coefs': list(coef[idx])}, columns=['words', 'coefs'])
    # dfsorted = newdf.sort_values('coefs')
    # plt.bar('words', 'coefs', data=dfsorted)
    # plt.show()




if __name__ == "__main__":
    main()
