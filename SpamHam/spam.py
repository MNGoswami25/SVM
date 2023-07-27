import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')
# import seaborn as sns # histogram plotting
ps = PorterStemmer()  # used to change dancing danced as only word  dance
#data loading




df = pd.read_csv('Spam.csv')

#print(df.sample(5))  # first five lines will be printed of dataset
#print(df.info())   # provides only data information  for cleaning


##########    DATA    CLEANING  ###################
# label encoding
from sklearn.preprocessing import LabelEncoder  # for giving label ham as 0 and spam as 1
encoder=LabelEncoder()   # LabelEncoder() ->  used to convert categorical labels into numeric format 0 for ham 1 for spam
df['Label'] = encoder.fit_transform(df['Label'])
#print(df.head())  # df.head()  is used to display few lines after label encoding

#misssing values
print(df.isnull().sum())  # used for checking any value is null or not

#duplicate value
print(df.duplicated().sum())    # we get 403 as duplicate items now we have to remove them

# remove dupluactes
df=df.drop_duplicates(keep='first')  # drop_duplicates  removes all duplicates
print(df.duplicated().sum())     # 0 printed

# print(df.shape)

############                   EDA-> Exploritary data analysis       ##################

# checking how much percent is spam and how much is ham by value_counts()
print(df['Label'].value_counts())
# for graphical representation of spam and ham
plt.pie(df['Label'].value_counts(),labels=['ham','spam'], autopct="%0.2f")
plt.show()   # will show graphical representation of data of spam and ham

# how many alphabets are used and how many words for that we have imported ntlk library


df['num_character'] = df['EmailText'].apply(len)  # we are counting length of emailtext and making a new coloumn as num_character and storing in it
#print(df.head)

df['num_words'] = df['EmailText'].apply(lambda x: len(nltk.word_tokenize(x)))  # splitting the text into words and counting its length

#print(df.head())
df['num_sentences']=df['EmailText'].apply(lambda x: len(nltk.sent_tokenize(x)))  # counting sentences

#print(df.sample(5))

###########################   DATA PRRPROCESSING ######################33
def transform_text(text):
    text = text.lower()   # change all data to lowercase
    text=nltk.word_tokenize(text)   # text into words

    y=[]
    for i in text:
        if i.isalnum():    # removing special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # removing stopwords  stopwords are like "you i my on "
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

print(transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."))
df['transformed_text'] = df['EmailText'].apply(transform_text)
#print(df.sample(5))

# model building

#text vectorization  email text to be converted into array form numerics  using back of words

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

cv=CountVectorizer()
#X = cv.fit_transform(df['transformed_text']).toarray()   #not using this because it giving less precision then tfidf
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['Label'].values
#print(x)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# precision and accuracy checking
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

# tfidf --> MNB

# we have imported these all algorithms  because we want to check which algos is giving good accuracy and precision
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# created the objects of above modules
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)



# created a dictionary in which keys are algos and value are objects
clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT':gbdt,

}

# function in which trainning and tseting part accuarcy and precision is getting
def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, precision

train_classifier(svc,X_train,y_train,X_test,y_test)

accuracy_scores = []   # array for accuracy and precision for holding value for every precision and accuracy and storing it to check which is better
precision_scores = []

#loop for same above accauracy and precision one

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# created a dataset of algorithmns and accuarcy and precision and in order so that highest comes first
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))