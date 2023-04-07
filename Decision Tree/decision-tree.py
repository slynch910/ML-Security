import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from numpy import mean
from numpy import std


onlineFilePath = "/content/drive/MyDrive/ML Security/Datasets/mirai_combined.csv"
kitsune = pd.read_csv(onlineFilePath, header=0)
feature_cols = kitsune.columns[1:-1]

X = kitsune[feature_cols]
Y = kitsune["Verdict"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

classifier = DecisionTreeClassifier(criterion="entropy", max_depth=1000, max_leaf_nodes=8, max_features=75)
classifier = classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

# Print out the metrics of the model being trained and tested.
print("Accuracy: ", metrics.accuracy_score(Y_test, y_pred))
print("========== Classification Report ==========")
print(metrics.classification_report(Y_test, y_pred, digits=3))
print("========== Confusion Matrix ==========")
print(metrics.confusion_matrix(Y_test, y_pred))
print("========== Tree Structure ==========")
print(tree.export_text(classifier))

num_splits = 15

# Now, make sure that this model isn't overfitted by performing cross-validation.
kf = KFold(n_splits=num_splits)
scorings = {
    'acc' : 'accuracy',
    'prec_macro': 'precision_macro',
    'rec_macro': 'recall_macro'
}

#scores = cross_val_score(classifier, X, Y, scoring='accuracy', cv=kf, n_jobs=-1)
#scores = cross_val_score(classifier, X, Y, scoring=scorings, cv=kf, n_jobs=-1)
scores = cross_validate(classifier, X, Y, scoring=scorings, cv=kf)


# Find statistics regarding the cross-validation.

print(scores.keys())
print(scores)

#print(scores["test_acc"])
print(len(scores))

print("=======================================================")
i = 0
for i in range(0,num_splits):
  print("Accuracy rating for k={}: {}".format(i+1, scores["test_acc"][i]))
  print("Precision rating for k={}: {}".format(i+1, scores["test_prec_macro"][i]))
  print("Recall rating for k={}: {}".format(i+1, scores["test_rec_macro"][i]))
  i += 1
  print("=======================================================")


