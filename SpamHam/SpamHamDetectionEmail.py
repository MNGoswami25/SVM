import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('Spam.csv')

print(data.info())

X = data['EmailText'].values
y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

classifier = SVC(kernel='rbf',random_state = 10)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))