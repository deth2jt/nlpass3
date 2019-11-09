
import glob
import codecs
import numpy
import re
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
#/Users/jthompson/Documents/nlp/ass3

src = [("neg.tok", "neg"), ("pos.tok","pos")]

def preprocess_text(text):
    #text = re.sub(' ,;()@-+>.\'','?"![][1-9]', text)
    text = re.sub('^ [A-Za-z]\n', '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def read_files(path):
    testfile = open(path, "r")
    testCurrfile = testfile.readlines()
    row = []
    for x in testCurrfile:
        x = x.strip('\n')

        x = preprocess_text(x)
        if(x != ''):
            row += [(path,x)]
    return row
     
def build_data_frame(path, c):
    row = []
    index = []

    for file, text in read_files(path):
        row.append({'text': text, 'class': c})
        index.append(file)
    dataFrame = DataFrame(row, index)
    print("dataFrame", dataFrame)
    return dataFrame

data = DataFrame()
for path, c in src:
    data = data.append(build_data_frame(path, c))


bowBinary = CountVectorizer( stop_words="english", lowercase = True, binary=True )
bowBinarynGram = CountVectorizer( stop_words="english", lowercase = True, binary=True, ngram_range=(1,2) )

pipeline = Pipeline([
            #('clf', MultinomialNB(alpha=1)) #Laplace smoothing, add-1	smoothing
            ('clf', BernoulliNB(alpha=1)) #Laplace smoothing, add-1	smoothing
                    ])

pipeline2 = Pipeline([
            ('clf', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs'))
                    ])

fold = KFold(10,shuffle=True, random_state=3)
scores=[]
confusion = numpy.array([[0,0],[0,0]])

X = data.loc[:,'text'].values

y = data.loc[:,'class'].values

print("	Logistic	Regression	classifier	with L2	regularization	using	binary bag-of-ngrams features (with	unigrams and	bigrams)")
for train_index, test_index in fold.split(X):
  
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]


   #	Logistic	Regression	classifier	with	L2	regularization	(and	default	parameters)	using	binary	bag-of-words	features
   
   X_train = bowBinarynGram.fit_transform(X_train)
   X_test = bowBinarynGram.transform(X_test)

   
   pipeline2.fit(X_train, y_train)
   preds = pipeline2.predict(X_test)


   

   confusion += confusion_matrix(y_test, preds)
   (tn, fp, fn, tp) = confusion.ravel()
   score = f1_score(y_test, preds, pos_label="pos")
   scores.append(score)

   print("len(data)", len(data))
   print("score", float(sum(scores)/len(scores)))
   print("confusion", confusion)

   print("\nTrue Negatives = " + str(tn))
   print("True Positives = " + str(tp))
   print("False Negatives = " + str(fn))
   print("False Positives = " + str(fp))

   actual_positives = tp+fn
   actual_negatives = tn+fp
   print("\nTotal Actual Positives = " + str(actual_positives))
   print("Total Actual Negatives = " + str(actual_negatives))

   print("\nTrue Positive Rate(TPR) = " + str(round(tp/actual_positives, 2)))
   print("True Negative Rate(TNR) = " + str(round(tn/actual_negatives, 2)))
   print("False Positive Rate(FPR) = " + str(round(fp/actual_negatives, 2)))
   print("False Negative Rate(FNR) = " + str(round(fn/actual_positives, 2)))
print("D:",  numpy.mean(scores))


scores=[]
confusion = numpy.array([[0,0],[0,0]])

print("Logistic	Regression	classifier	with	L2	regularization	(and default	parameters)	using	binary	bag-of-words	features")
for train_index, test_index in fold.split(X):
  
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   #	Logistic	Regression	classifier	with	L2	regularization	using	binary	bag-of-ngrams	features
   
   X_train = bowBinary.fit_transform(X_train)
   X_test = bowBinary.transform(X_test)

   
   pipeline2.fit(X_train, y_train)
   preds = pipeline2.predict(X_test)


   

   confusion += confusion_matrix(y_test, preds)
   (tn, fp, fn, tp) = confusion.ravel()
   score = f1_score(y_test, preds, pos_label="pos")
   scores.append(score)

   print("len(data)", len(data))
   print("score", float(sum(scores)/len(scores)))
   print("confusion", confusion)

   print("\nTrue Negatives = " + str(tn))
   print("True Positives = " + str(tp))
   print("False Negatives = " + str(fn))
   print("False Positives = " + str(fp))

   actual_positives = tp+fn
   actual_negatives = tn+fp
   print("\nTotal Actual Positives = " + str(actual_positives))
   print("Total Actual Negatives = " + str(actual_negatives))

   print("\nTrue Positive Rate(TPR) = " + str(round(tp/actual_positives, 2)))
   print("True Negative Rate(TNR) = " + str(round(tn/actual_negatives, 2)))
   print("False Positive Rate(FPR) = " + str(round(fp/actual_negatives, 2)))
   print("False Negative Rate(FNR) = " + str(round(fn/actual_positives, 2)))
print("C:",  numpy.mean(scores))




scores=[]
confusion = numpy.array([[0,0],[0,0]])
print("	A	Naïve	Bayes	classifier	with	add-1	smoothing	using	binary bagof-words	features")
for train_index, test_index in fold.split(X):
  
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   #	Logistic	Regression	classifier	with	L2	regularization	using	binary	bag-of-ngrams	features
   
   X_train = bowBinary.fit_transform(X_train)
   X_test = bowBinary.transform(X_test)

   
   pipeline.fit(X_train, y_train)
   preds = pipeline.predict(X_test)


   

   confusion += confusion_matrix(y_test, preds)
   (tn, fp, fn, tp) = confusion.ravel()
   score = f1_score(y_test, preds, pos_label="pos")
   scores.append(score)

   print("len(data)", len(data))
   print("score", float(sum(scores)/len(scores)))
   print("confusion", confusion)

   print("\nTrue Negatives = " + str(tn))
   print("True Positives = " + str(tp))
   print("False Negatives = " + str(fn))
   print("False Positives = " + str(fp))

   actual_positives = tp+fn
   actual_negatives = tn+fp
   print("\nTotal Actual Positives = " + str(actual_positives))
   print("Total Actual Negatives = " + str(actual_negatives))

   print("\nTrue Positive Rate(TPR) = " + str(round(tp/actual_positives, 2)))
   print("True Negative Rate(TNR) = " + str(round(tn/actual_negatives, 2)))
   print("False Positive Rate(FPR) = " + str(round(fp/actual_negatives, 2)))
   print("False Negative Rate(FNR) = " + str(round(fn/actual_positives, 2)))
print("A:", numpy.mean(scores))


scores=[]
confusion = numpy.array([[0,0],[0,0]])
print("A Naïve Bayes classifier	with add-1	smoothing	using	binary	bagof-ngrams	features	(with	unigrams	and	bigrams).	")
for train_index, test_index in fold.split(X):
  
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]


   #	Logistic	Regression	classifier	with	L2	regularization	(and	default	parameters)	using	binary	bag-of-words	features
   
   X_train = bowBinarynGram.fit_transform(X_train)
   X_test = bowBinarynGram.transform(X_test)

   
   pipeline.fit(X_train, y_train)
   preds = pipeline.predict(X_test)


   

   confusion += confusion_matrix(y_test, preds)
   (tn, fp, fn, tp) = confusion.ravel()
   score = f1_score(y_test, preds, pos_label="pos")
   scores.append(score)

   print("len(data)", len(data))
   print("score", float(sum(scores)/len(scores)))
   print("confusion", confusion)

   print("\nTrue Negatives = " + str(tn))
   print("True Positives = " + str(tp))
   print("False Negatives = " + str(fn))
   print("False Positives = " + str(fp))

   actual_positives = tp+fn
   actual_negatives = tn+fp
   print("\nTotal Actual Positives = " + str(actual_positives))
   print("Total Actual Negatives = " + str(actual_negatives))

   print("\nTrue Positive Rate(TPR) = " + str(round(tp/actual_positives, 2)))
   print("True Negative Rate(TNR) = " + str(round(tn/actual_negatives, 2)))
   print("False Positive Rate(FPR) = " + str(round(fp/actual_negatives, 2)))
   print("False Negative Rate(FNR) = " + str(round(fn/actual_positives, 2)))
print("B:",  numpy.mean(scores))
   
