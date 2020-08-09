from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

def decision_tree(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)

    prediction = clf.predict(x_test).copy()
    print('=======================')
    print('Predicted Labels for Decision Tree')
    print('=======================')

    #Question 3.2
    accuracy = accuracy_score(prediction,y_test)
    print('=======================')
    print('Accuracy for Decision Tree')
    print('=======================')
    print(accuracy)
    #Question 3.3
    cm = confusion_matrix(prediction,y_test)
    print('=======================')
    print('Confusion Matrix for Decision Tree')
    print('=======================')
    print(cm)

    tp = cm[0][0] # True positive

    fn = cm[1][0] # False Negative

    fp = cm[0][1] # False Positive

    tn = cm[1][1] # True negative

    tpr = tp/(tp+fn) # True Positive Rate
    tnr = tn/(tn + fp) # True negative Rate

    print("===================================")
    print('True Positives - ' + str(tp))
    print('False Negatives - ' + str(fn))
    print('False Positives - ' + str(fp))
    print('True Negatives - ' + str(tn))
    print("===================================")
    print('True Positive Rate - ' + str(round(tpr,3)))
    print('True Negative Rate - ' + str(round(tnr,3)))