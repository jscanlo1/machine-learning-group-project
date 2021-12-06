#yesdocs array of strings
#nodocs


from keras.backend import dropout
import nltk
import collections
import json
import pandas as pd
from collections import Counter
from numpy.core.fromnumeric import size
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, Embedding, Input
nltk.download('punkt')
nltk.download('wordnet')


def ratingToNum(ratingString):
    if(ratingString==('no factual content')):
        return 1
    elif(ratingString==('mostly false')):
        return -1
    elif(ratingString==('mixture of true and false')):
        return -1
    elif(ratingString==('mostly true')):
        return 1
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
    f = open('AllComments.json',encoding="utf8")

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

    print(words[:10])

    import matplotlib.pyplot as plt

    plt.imshow(tfidf)
    plt.show()
    import numpy as np

    x = tfidf
    y = df['rating']
    y = np.array(y)
    print(x.shape)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    
    #Imports for F1 score and ROC curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import confusion_matrix

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = LogisticRegression(penalty="l2", C=1).fit(x_train, y_train)

    #print(len(model.predict_proba(x_test)[1]))
    y_pred = model.predict(x_test)
    #y_predict = [int(p[1] > 0.5) for p in model.predict_proba(x_test)]
    print(model.coef_.shape)

    coef = model.coef_.reshape(-1)
    print(coef.shape)
    np.argmax(coef)
    #plt.plot(words, coef[3], label='Mostly True')
    plt.plot(words, coef, label='Mostly True')
    #plt.plot(words, coef[0], label='Mostly False')
    plt.legend()
    plt.xticks(words, words, rotation='vertical')
    plt.show()
    idx = np.argsort(coef)[-10:]
    print(idx)
    words = np.array(words)

    print(words[idx])

    print("---------------Logistic-----------")
    print(classification_report(y_test, y_pred)) 
    print(confusion_matrix(y_test,y_pred))
    print(f1_score(y_test,y_pred))
    probs_logistic = model.predict_proba(x_test)


    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="stratified").fit(x_train, y_train)
    y_dummy = dummy.predict(x_test)
    print("---------------Dummy Classifier (Stratified)-----------")
    print(classification_report(y_test, y_dummy))
    print(confusion_matrix(y_test,y_dummy))
    print(f1_score(y_test,y_dummy))
    probs_dummy = dummy.predict_proba(x_test)


    from sklearn.svm import LinearSVC
    model_SVM_L1 = LinearSVC(C=0.1, penalty="l1",dual=False)
    model_SVM_L1.fit(x_train, y_train)
    y_preds_SVM_L1 = model_SVM_L1.predict(x_test)
    print("---------------Linear SVM L1-----------")
    print(classification_report(y_test, y_preds_SVM_L1))
    print(confusion_matrix(y_test,y_preds_SVM_L1))
    print(f1_score(y_test,y_preds_SVM_L1))

    from sklearn.calibration import CalibratedClassifierCV
    clf = CalibratedClassifierCV(model_SVM_L1) 
    clf.fit(x_train, y_train)
    y_proba_L1 = clf.predict_proba(x_test)
    #probs_SVM = model_SVM.predict_proba(x_test)


    model_SVM_L2 = LinearSVC(C=0.1, penalty="l2")
    model_SVM_L2.fit(x_train, y_train)
    y_preds_SVM_L2 = model_SVM_L2.predict(x_test)
    print("---------------Linear SVM L2-----------")
    print(classification_report(y_test, y_preds_SVM_L2))
    print(confusion_matrix(y_test,y_preds_SVM_L2))
    print(f1_score(y_test,y_preds_SVM_L2))

    from sklearn.calibration import CalibratedClassifierCV
    clf = CalibratedClassifierCV(model_SVM_L2) 
    clf.fit(x_train, y_train)
    y_proba_L2 = clf.predict_proba(x_test)



    
    from sklearn.svm import SVC
    model_SVC = SVC(C=0.1, kernel='rbf', gamma=5).fit(x_train, y_train)
    model_SVC.fit(x_train, y_train)
    y_preds_SVC = model_SVC.predict(x_test)
    print("---------------kernalised SVC-----------")
    print(classification_report(y_test, y_preds_SVC))
    print(f1_score(y_test,y_preds_SVC))

    from sklearn.calibration import CalibratedClassifierCV
    clf = CalibratedClassifierCV(model_SVC) 
    clf.fit(x_train, y_train)
    y_proba_SVC = clf.predict_proba(x_test)



    #BEGIN DEALING WITH DEEP LEARNING
    num_classes = 2
    input_shape = (1, 622)

    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OrdinalEncoder
    #oe = OrdinalEncoder()
    #oe.fit(x_train)
    #x_train_enc = oe.transform(x_train)
    #x_test_enc = oe.transform(x_test)

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    #CHANGE INPUT SHAPES FOR CONV1D.
    #Comment out if using dense layer model

   # x_train_DL = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test_DL = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    #Input for Dense layer
    #Comment out if using Conv1D
    x_train_DL = x_train
    x_test_DL = x_test

    model_DL = keras.Sequential()

    '''
    #Conv1d model
    model_DL.add(Input(x_train_DL.shape[1:]))
    model_DL.add(Conv1D(32, 5, padding='same', activation='relu'))
    model_DL.add(Conv1D(32, 5, padding='same' ,activation='relu'))
    model_DL.add(Dropout(0.5))
    #model_DL.add(MaxPooling1D())
    model_DL.add(Conv1D(64, 3, padding='same' ,activation='relu'))
    model_DL.add(Conv1D(64, 3, padding='same' ,activation='relu'))
    model_DL.add(Dropout(0.5))
    model_DL.add(Flatten())
    model_DL.add(Dense(10, activation='relu'))
    model_DL.add(Dense(1, activation='sigmoid'))
    '''

    #dense layer model
    model_DL.add(Dense(100, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
    #model_DL.add(Dropout(0.5))
    #model_DL.add(Dense(10, activation='relu'))
    model_DL.add(Dense(1, activation='sigmoid'))


    model_DL.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    model_DL.summary()


    #batch_size = 16
    epochs = 40
    #history = model_DL.fit(x_train_DL, y_train_enc, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    history = model_DL.fit(x_train_DL, y_train_enc, epochs=epochs, validation_split=0.1)
	#model_DL.save("cifar.model")
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


    y_pred_DL = model_DL.predict(x_test_DL)
    #y_pred_DL_1 = np.argmax(y_pred_DL, axis=1)
    #y_test1_DL = np.argmax(y_test, axis=1)
    y_pred_DL_1 =(y_pred_DL>0.5)
    y_test_DL =(y_test_enc>0.5)
    #np.reshape(y_test_DL, (len(y_pred_DL_1),1)).T
    list(y_pred_DL_1)
    list(y_test_DL)

    #print(y_pred_DL_1)
    #print(y_test_DL)
    #print(classification_report(y_test_DL, y_pred_DL))
    print("---------------Multi-Layer Perceptron-----------")
    print(classification_report(y_test_DL, y_pred_DL_1))
    print(confusion_matrix(y_test_DL,y_pred_DL_1))
    print(f1_score(y_test_DL,y_pred_DL_1))

    #y_pred_keras = model_DL.predict(x_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_DL)
    roc_auc_keras = auc(fpr_keras, tpr_keras)










    
    


    fpr_logistic, tpr_logistic, threshold = roc_curve(y_test, probs_logistic[:,1])
    roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

    fpr_dummy, tpr_dummy, threshold = roc_curve(y_test, probs_dummy[:,1])
    roc_auc_dummy = auc(fpr_dummy, tpr_dummy)


    fpr_SVM_L1, tpr_SVM, threshold = roc_curve(y_test, y_proba_L1[:,1])
    roc_auc_SVM_L1 = auc(fpr_SVM_L1, tpr_SVM)

    fpr_SVM_L2, tpr_SVM_L2, threshold = roc_curve(y_test, y_proba_L2[:,1])
    roc_auc_SVM_L2 = auc(fpr_SVM_L2, tpr_SVM_L2)

    fpr_SVC, tpr_SVC, threshold = roc_curve(y_test, y_proba_SVC[:,1])
    roc_auc_SVC = auc(fpr_SVC, tpr_SVC)




    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_dummy, tpr_dummy, 'r', label = 'Dummy - AUC = %0.2f' % roc_auc_dummy)
    plt.plot(fpr_SVM_L1, tpr_SVM, 'b', label = 'SVM (L1) - AUC = %0.2f' % roc_auc_SVM_L1)
    plt.plot(fpr_SVM_L1, tpr_SVM, 'y', label = 'SVM (L2) - AUC = %0.2f' % roc_auc_SVM_L2)
    #plt.plot(fpr_SVC, tpr_SVC, 'r', label = 'SVM with kernel - AUC = %0.2f' % roc_auc_SVC)
    plt.plot(fpr_logistic,tpr_logistic, color='green',linestyle='--', label = 'Logistic - AUC = %0.2f' % roc_auc_logistic)
    plt.plot(fpr_keras,tpr_keras, color='m',linestyle='--', label = 'Multi-Layer Perceptron - AUC = %0.2f' % roc_auc_keras)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves')
    plt.legend(loc = 'lower right')
    plt.show()












if __name__ == "__main__":
    main()
