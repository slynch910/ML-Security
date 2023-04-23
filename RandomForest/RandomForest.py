import numpy as np, sklearn
from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree   
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import sys, csv, threading

from numpy.lib import average
import numpy as np

import matplotlib.pyplot as plt


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=2, random_state=0, num_folds=10, cv=10):
        """ Initialize the Random Forest model. """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.num_folds = num_folds
        self.cv = cv
        
    def train(self):
        """ Train the Random Forest model. """
        print(f"[+] Training the Random Forest model on {type(self).__name__} dataset...")
        # creating a RF classifier
        # self.model =  RandomForestClassifier(
        #     n_estimators=self.n_estimators, 
        #     max_depth=self.max_depth, 
        #     random_state=self.random_state
        # )  

        self.pick_hyperparams()
        
        # Training the model on the training dataset fit function is used to train the model using the training sets as parameters
        # self.model.fit(self.x_train, self.y_train)
        
        # performing predictions on the test dataset
        self.y_pred = self.model.predict(self.x_test)

    def metrics(self):
        """ Evaluate the Random Forest model. """
        print(f"[+] Evaluating the Random Forest model on {type(self).__name__} dataset...")

        # print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(self.y_test, self.y_pred))

        # Predicting on the test data.
        accuracy =f"Accuracy: {accuracy_score(self.y_test, self.y_pred)}"
        report = f"========= Classification Report ========= \n {classification_report(self.y_test, self.y_pred, digits=6)}"

        print(accuracy)
        print(report)

        print(f"[*] Saving metrics for {type(self).__name__} dataset...")
        with open(f"results/{type(self).__name__}_metrics.txt", "w") as out:
            out.write(accuracy)
            out.write(report)

        print(f"[+] Creating Confusion Matrix for {type(self).__name__} dataset...")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, self.y_pred), 
            # display_labels=self.Y.unique()
        ).plot()

        plt.savefig(f"results/{type(self).__name__}_confusion_matrix.png") 
        # plt.show()
     
    def cross_validate(self): 
        """ Cross validate the Random Forest model. """
        print("[+] Cross validating the Random Forest model...") 
        
        # cross_validate also allows to specify metrics which you want to see
        scores = cross_validate(
            self.model, 
            self.X, 
            self.Y, 
            scoring={
                'acc' : 'accuracy',
                'prec_macro': 'precision_macro',
                'rec_macro': 'recall_macro'
            }, 
            cv=StratifiedKFold(n_splits=self.num_folds,random_state=42,shuffle=True), 
            return_train_score=True
        )  
        
        # print(scores)
        # for i, score in enumerate(scores["test_score"]):
        #     print(f"Accuracy for the fold no. {i} on the test set: {score}")

        # Print out the results of the cross validation for each fowld. 
        results = []
        print(f"Cross validation results for {type(self).__name__} dataset:")
        print("=======================================================")
        for i in range(0,self.num_folds):
            for key in scores.keys():
                print(key)

            results.append((
                scores["test_acc"][i], 
                scores["test_prec_macro"][i], 
                scores["test_rec_macro"][i]
            ))            
            
            print(f"Accuracy rating for k={i+1}: {scores['test_acc'][i]}")  
            print(f"Precision rating for k={i+1}: {scores['test_prec_macro'][i]}")
            print(f"Recall rating for k={i+1}:{scores['test_rec_macro'][i]}") 
            print("=======================================================")

        # print(f"Mean Test Accuracy: {np.lib.average(scores['test_acc'])}")
        # print(f"Mean Test Precision: {np.lib.average(scores['test_prec_macro'])}")
        # print(f"Mean Test Recall: {np.lib.average(scores['test_rec_macro'])}")

        print(f"[*] Saving cross validation results for {type(self).__name__} dataset...")
        with open(f"results/{type(self).__name__}_results.csv", "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["Accuracy", "Precision", "Recall"])

            # csv_out.writerow([
            #     np.lib.average(scores["test_acc"]), 
            #     np.lib.average(scores["test_prec_macro"], 
            #     np.lib.average(scores["test_rec_macro"]))
            # ])
            
            csv_out.writerows(results)

    def pick_hyperparams(self):
        """ Pick the best hyperparameters for the Random Forest model. """

        print(f"[+] Picking the best hyperparameters for the Random Forest model on {type(self).__name__} dataset...")

        # Initialize a cross-validation fold and perform a randomized-search to tune the hyperparameters

        # Instantiate RandomizedSearchCV model
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_jobs=-1, random_state=42),
            param_distributions= {
                "n_estimators": np.arange(10, 100, 10), 
                "max_depth": [None, 3, 5, 10],
                "min_samples_split": np.arange(2, 20, 2),
                "min_samples_leaf": np.arange(1, 20, 2),
                "max_features": [0.5, 1, "sqrt"],
                "max_samples": [10000]
            }, 
            n_iter=2,
            cv=self.cv        
        )
                
        search_results = random_search.fit(self.x_train, self.y_train)

        # extract the best model and evaluate it
        print("[*] Picked best hyperparms, now evaluating...")
        
        best_models = search_results.best_estimator_
        best_params = search_results.best_params_ 
        
        print(f"R2: {best_models.score(self.x_train, self.y_train)}")
        print(f"Best params: {best_params}")

        self.model = best_models

        # Write best params to file in results directory.
        with open(f"results/{type(self).__name__}_best_params.txt", "w") as out:
            out.write(str(best_params))

    def run(self): 
        """ Run the Random Forest model."""
        self.train()
        self.metrics()
        self.cross_validate()

        print(f"[+] Done with {type(self).__name__} ")

class KDD(RandomForest):
    def __init__(self, dataset_location="datasets/KDDTrain+.arff", *a, **k):
        """ Initialize the Random Forest model. """
        super().__init__(*a, **k)

        # Load model into DataFrame from ARFF file.
        self.dataset = pd.DataFrame(arff.loadarff(dataset_location)[0])
    
        # Clean the data and fix mappings 
        self.clean_data() 

        # Split the data into training and testing sets.
        self.X, self.Y = self.dataset.loc[:, self.dataset.columns != "class"], self.dataset["class"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)

    def clean_data(self): 
        for cols in ["protocol_type", "service", "flag", "class"]: 
            self.dataset[cols] = self.dataset[cols].str.decode("utf-8")

        self.dataset = self.dataset.replace({"protocol_type": {"tcp": 0, "udp": 1, "icmp": 2,}})
        self.dataset = self.dataset.replace({"class": {"normal": 0, "anomaly": 1}})

        categories_that_need_mappings = ["service", "flag"]
        for category in categories_that_need_mappings: 
            mappings = {}
            for i, value in enumerate(list(set(self.dataset[category]))):
                mappings.update({value: i}) 

            self.dataset = self.dataset.replace({category: mappings})

class MIRAI(RandomForest): 
    def __init__(self, dataset_location="datasets/mirai_combined.csv", *a, **k):
        """ Initialize the Random Forest model for MIRAI. """
        super().__init__(*a, **k)

        # Load model into DataFrame from ARFF file.
        self.dataset = pd.DataFrame(pd.read_csv(dataset_location))
        
        # Split the data into training and testing sets.
        self.X, self.Y = self.dataset[self.dataset.columns[1:-1]], self.dataset["Verdict"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=0)

class UCI(RandomForest): 
    def __init__(self, dataset_location="datasets/uci_dataset.csv", *a, **k):
        """ Initialize the Random Forest model for UCI. """
        super().__init__(*a, **k)

        # Load model into DataFrame from ARFF file.
        self.dataset = pd.DataFrame(pd.read_csv(dataset_location))
        self.dataset.head()

        # Split the data into training and testing sets.
        self.X, self.Y = self.dataset[self.dataset.columns[0:115]], self.dataset["Label"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=42)

class IOT(RandomForest): 
    def __init__(self, dataset_location="datasets/iot_23_dataset.csv", *a, **k):
        """ Initialize the Random Forest model for IOT. """
        super().__init__(*a, **k)

        # Load model into DataFrame from ARFF file.
        self.dataset = pd.DataFrame(pd.read_csv(dataset_location))
        
        # Split the data into training and testing sets.
        self.X, self.Y = self.clean_data()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=42) 

    def clean_data(self):
        p = LabelEncoder()
        self.dataset.head()

        #feature scaling because several of the columns are strings 
        self.dataset['id.orig_h'] = p.fit_transform(self.dataset['id.orig_h'])
        self.dataset['id.resp_h'] = p.fit_transform(self.dataset['id.resp_h'])
        self.dataset['proto'] = p.fit_transform(self.dataset['proto'])
        self.dataset['conn_state'] = p.fit_transform(self.dataset['conn_state'])
        self.dataset['history'] = p.fit_transform(self.dataset['history'])
        self.dataset['label'] = p.fit_transform(self.dataset['label']) 
        self.dataset = self.dataset.drop(columns=['uid', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp'])
        self.dataset.head()

        X = self.dataset[self.dataset.columns[0:13]]
        Y = self.dataset["label"]

        return X, Y

if __name__ == "__main__": 

    print(f"[+] Starting {sys.argv[1].lower()} ")

    match sys.argv[1].lower():

        case "kdd":
            KDD().run()
        case "mirai":
            MIRAI().run()
        case "iot":
            IOT().run()
        case "uci":
            UCI().run()

        case "all":
            models = [KDD(), MIRAI(), IOT(), UCI()]
            map(lambda x: x.run(), models)    

        case "thread":
            models = [KDD(), MIRAI(), IOT(), UCI()]
            map(lambda x: threading.Thread(target=x.run).start(), models)    

        case _:
            print("[-] Invalid dataset name. Please choose from: kdd, mirai, iot, uci")
            sys.exit(0)
