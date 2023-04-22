import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv




####################################################
# Class to hold the DT for the IoT data
####################################################
class DT:
    
    # Number of folds for cross validation
    num_folds = 15

    # Scoring Dictionary
    scoring_dict = {
        'acc' : 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro'
    }
    
    # Columns to exclude since they are not useful and some entries don't have them.
    EXCLUDE_COLS = ["uid", "service", "duration", "orig_bytes", "resp_bytes", "local_orig", "local_resp"]
    
    # Columns to scale since they are mainly strings. 
    FEATURE_SCALING_COLS = ["id.orig_h", "id.resp_h", "proto", "conn_state", "history", "label"]
    
    ####################################################
    # Constructor
    ####################################################    
    def __init__(self, datasetPath="iot_23_dataset.csv"):
        
        # Read in the data.
        self.dataset = pd.read_csv(datasetPath, header=0)
        
        # Remove the columns that are not useful.
        for col in self.EXCLUDE_COLS:
            self.dataset.drop(col, axis=1, inplace=True)
        
        # Create an encoder.
        encoder = LabelEncoder()
        
        # Convert the strings of the columns to numbers.
        for feature in self.FEATURE_SCALING_COLS:
            self.dataset[feature] = encoder.fit_transform(self.dataset[feature])
            
        
        print(self.dataset.head())
        
        # Split the data into data and labels
        # Data
        self.X = self.dataset[self.dataset.columns[:-1]]
        # Labels
        self.Y = self.dataset[self.dataset.columns[-1]]
        
        # Split the data into training and testing data.
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=1)
        
    ####################################################
    # Train the model with the training data.
    ####################################################
    def train(self):
        
        # Create the decision tree classifier.
        self.classifier = DecisionTreeClassifier(criterion="entropy", max_depth=32, max_leaf_nodes=6, max_features=7)
        
        # Train the model.
        self.classifier = self.classifier.fit(self.X_train, self.Y_train)
        
        # Predict the labels for the test data.
        self.y_pred = self.classifier.predict(self.X_test)
        
        # Print out the metrics from the model.
        self.print_metrics()
        
        # Perform cross validation on the model.
        self.perform_cross_validation()
    
    
    ####################################################
    # Print out the metrics for the model.
    ####################################################
    def print_metrics(self):
        print("accuracy: ", metrics.accuracy_score(self.Y_test, self.y_pred))
        print("========== Classification Report ==========")
        print(metrics.classification_report(self.Y_test, self.y_pred, digits=3))
        print("========== Confusion Matrix ==========")
        print(metrics.confusion_matrix(self.Y_test, self.y_pred))   
        print("========== Tree Structure ==========")
        print(tree.export_text(self.classifier))
        
    ##############################################
    # Perform cross validation and print results.
    ##############################################
    def perform_cross_validation(self):
        
        # Create the Kfold object for cross validation
        kf = KFold(n_splits=self.num_folds)
        
        # Perform the cross validation
        scores = cross_validate(self.classifier, self.X, self.Y, scoring=self.scoring_dict, cv=kf, return_train_score=True)

        results = []
        
        # Print out the results of the cross validation for each fold. 
        print("=======================================================")
        for i in range(0,self.num_folds):
          results.append([scores["test_acc"][i], scores["test_prec_macro"][i], scores["test_rec_macro"][i]])
          print("Accuracy rating for k={}: {}".format(i+1, scores["test_acc"][i]))
          print("Precision rating for k={}: {}".format(i+1, scores["test_prec_macro"][i]))
          print("Recall rating for k={}: {}".format(i+1, scores["test_rec_macro"][i]))
          print("=======================================================")


        with open("dt_results.csv", "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["Accuracy", "Precision", "Recall"])
            csv_out.writerows(results)

    
if __name__ == "__main__":
    dt = DT()
    dt.train()
    