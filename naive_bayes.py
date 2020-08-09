from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

def decision_tree(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train_dt, y_train_dt)
    
    prediction_dt = clf.predict(x_test_dt).copy()
    print('=======================')
    print('Predicted Labels for Decision Tree')
    print('=======================')
    print(prediction_dt)
    
    #Question 3.2
    accuracy_dt = accuracy_score(prediction_dt,y_test_dt)
    print('=======================')
    print('Accuracy for Decision Tree')
    print('=======================')
    print(accuracy_dt)
    #Question 3.3
    confusion_matrix_dt = confusion_matrix(prediction_dt,y_test_dt)
    print('=======================')
    print('Confusion Matrix for Decision Tree')
    print('=======================')
    print(confusion_matrix_dt)