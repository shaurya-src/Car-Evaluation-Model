from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import sklearn

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

X = list(zip(buying, maint, door, persons, lug_boot, safety, cls))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print('Accuracy = ', acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted : ", names[predicted[x]], "Data : ", x_test[x], "Actual : ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 5, True)
    print("N : ", n)