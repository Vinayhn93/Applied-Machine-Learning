
'''
Mail Spam Detection Using Naive Bayes
Objective: To determine the given mail is Spam or not Spam
Method/Model : Naive Bayes
Result: We achieved 98% of the Accuracy in Our Model
Reference : https://towardsdatascience.com/naive-bayes-intuition-and-implementation-ac328f9c9718
'''
import pandas as pd
#Data Source : https://archive.ics.uci.edu/ml/machine-learning-databases/00228/
Data = pd.read_table('C:\\Users\\HP Intel i5\\Desktop\\Applied AI\\Naive1',names=['Label','Message'])

'''
1.Data Preprocessing

    Convert Spam as 1 and ham as 0
'''

Data.Label = Data.Label.map({'ham':0,'spam':1})
Document = Data.Message

"How Many times the word occur in a Document"
"Split the Data Set into training and Test Data"


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(Data['Message'], Data['Label'], random_state=50)


Count_Vector = CountVectorizer()
training_data = Count_Vector.fit_transform(X_train)
testing_data = Count_Vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score
print('Accuracy score:{} %'.format(accuracy_score(y_test, predictions)*100))