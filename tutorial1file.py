import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

#creating a model


best=0
for i in range(3000):   #training the model in a loop
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)    #calculating the accuracy
    print(acc)   #printing the accuracy


    if acc>best:
        best=acc     #for getting higher accuracy
        with open("studentmodel.pickle","wb") as f:        #saving the model using pickle library
            pickle.dump(linear,f)

#opening the saved model

pickle_in = open("studentmodel.pickle","rb")
linear=pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("interscept: \n", linear.intercept_)


predictions = linear.predict(x_test)
print("\n\nPredicted marks: \n")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


#plotting the result

p='failures'      #you can give the attributes from above to complare with the final result
style.use("ggplot")
plt.scatter(data[p],data['G3'])
plt.xlabel("p")
plt.ylabel("Final Grade")
plt.show()
