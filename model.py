import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('garments_worker_productivity.csv')

#df


df.isnull().sum()

df=df.interpolate()



#df

df.isnull().sum()

prod_filter=df['actual_productivity']>=1
prod_filter.sum()

prod1_filter=df['actual_productivity']<=1

df=df.loc[prod1_filter, :]


df.isnull().sum()

#df

df.describe()

df.info()

# Identify and address any duplicates
df.duplicated().sum()

#import seaborn as sns
#import matplotlib.pyplot as plt

#sns.heatmap(df.corr(), cmap='Reds')

#plt.figure(figsize = (10,5))
#sns.boxplot(data = df)
#plt.xticks(rotation = 90)
#plt.title("Box Plot")


plt.figure(figsize=(10,5))
sns.countplot(x= 'department',hue='quarter',data=df)

df['department'].value_counts().plot.pie()

df.drop(['date'],axis=1, inplace=True)

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
for column in df.columns:
    df[column]= label_encoder.fit_transform(df[column])
#df

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeRegressor 

dtr = DecisionTreeRegressor()

model=dtr.fit(X_train, y_train)
y_pred_lreg = dtr.predict(X_test)
print(y_pred_lreg)
print(y_test)
logreg_accuracy = round(accuracy_score(y_test, y_pred_lreg)*100,2)
print('Accuracy', logreg_accuracy, '%')


predictions = dtr.predict(X_test)
true_labels = y_test

#cf_matrix = confusion_matrix(true_labels, predictions)
#plt.figure(figsize = (7,4))
#heatmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt = 'g',
#                     xticklabels=np.unique(true_labels),
#                     yticklabels=np.unique(true_labels))

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
 

