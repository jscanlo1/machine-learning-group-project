import collections
import json
from collections import Counter

f = open('ALLComments.json',encoding="utf8")

articles = json.load(f)

dictionary = {}

def ratingToNum(ratingString):
    if(ratingString==('no factual content')):
        return 2
    elif(ratingString==('mostly false')):
        return 1
    elif(ratingString==('mixture of true and false')):
        return 1
    elif(ratingString==('mostly true')):
        return 2
    else:
        print("OTHER RATING: ", ratingString)
    


class post:
    def __init__(self, ID, URL, RATING):
        self.id = ID
        self.url = URL
        self.rating = ratingToNum(RATING)
        self.words = []
        self.comments = ''

    def addWords(self, WORDS):
        self.words += WORDS

    def addComments(self, COMMENT):
        #self.comments.append(COMMENT + ' ')
        self.comments += COMMENT + ' '

    def addDict(self, dictCopy):
        self.dict = dictCopy.copy()
        for wordToAdd in self.words:
            if (self.dict).get(wordToAdd) != None:
                self.dict[wordToAdd]  = self.dict[wordToAdd] + 1
            else:
                print("ERROR: NEW WORD IN DICT")

    def print_all(self):
        print(f"ID: {self.id}\tURL: {self.url}\tRATING: {self.rating}")
        print(f"COMMENTS: {self.words}")
        #print(f"DICT: {self.dict}")

    def print_dict(self):
        for key, entry in ((self.dict).items()):
            print(f"Word: {key}\t\Instances: {entry}")
        #print(f"DICT: {self.dict}")

    def ratingAndDictAndWords(self):
        return self.rating, self.dict, self.words
        #print(f"DICT: {self.dict}")

    def ratingAndDictAndComments(self):
        return self.rating, self.dict, self.comments
        #print(f"DICT: {self.dict}")

    def get_comments(self):
        return self.comments
        #print(f"DICT: {self.dict}")
        


postid = 0
postList = []

for article in articles:
    postURL = article['PostURL']
    rating = article['TruthRating']
    comments = article['Comments']

    currPost = post(postid, postURL, rating)

    postid = postid + 1
    #print(ratingToNum(rating))
    for comment in comments:
        currPost.addComments(comment)
        words = comment.split()
        currPost.addWords(words)
        for word in words:
            if(dictionary.get(word) == None):
                dictionary[word] = 1
            else:
                dictionary[word] += 1

    #print(currPost.get_comments())
    print(currPost.id)
    postList.append(currPost)
    del currPost

sortedDict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
#dictCopy = sortedDict.copy()
dictCopy = {k: 0 for k, v in sorted(dictionary.items(), key=lambda item: item[1])}

for postX in postList:
    postX.addDict(dictCopy)



#print(sortedDict)

#print(len(postList))
#for key, entry in (dictCopy.items()):
#    print(f"Word: {key}\t\Instances: {entry}")

#postList[0].print_all()
postList[1].print_all()
#postList[1].print_dict()

#from sklearn.datasets import load_files
#d = load_files(”txt_sentoken”, shuffle=False)
x=[]
y=[]
for postX in postList:
    rat, dic, com = postX.ratingAndDictAndComments()
    x.append(com)
    y.append(rat)

    #print(com)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',lowercase='True' ,max_df=0.2)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
Xtrain = vectorizer.fit_transform(xtrain)

print(Xtrain)

Xtest = vectorizer.transform(xtest)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
model = LinearSVC(C=0.5)
model.fit(Xtrain, ytrain)
preds = model.predict(Xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest, preds))
from sklearn.metrics import f1_score
print(f1_score(ytest,preds))

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
ydummy = dummy.predict(Xtest)
print(classification_report(ytest, ydummy))
print(f1_score(ytest,ydummy))
