import numpy as np, sklearn
from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree   
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import sys, csv 

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
        print("Accuracy: ", accuracy_score(self.y_test, self.y_pred))
        print("========= Classification Report =========")
        print(classification_report(self.y_test, self.y_pred, digits=6))
        print("========= Confusion Matrix =========")
        print(confusion_matrix(self.y_test, self.y_pred))
        # print("========= Tree Structure =========")
        # print(tree.export_text(self.model))

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
            cv=self.cv, 
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
    
        print(f"[*] Saving cross validation results for {type(self).__name__} dataset...")
        with open(f"results/{type(self).__name__}_results.csv", "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["Accuracy", "Precision", "Recall"])
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

    match sys.argv[1].lower():
        case "kdd":
            rf = KDD()
        case "mirai":
            rf = MIRAI()
        case "iot":
            rf = IOT()
        case "uci":
            rf = UCI()
        case _:
            print("[-] Invalid dataset name. Please choose from: kdd, mirai, iot, uci")
            sys.exit(0)

    rf.train()
    rf.metrics()
    rf.cross_validate()



"""

ipython RandomForest.py KDD
[+] Training the Random Forest model on KDD dataset...
[+] Picking the best hyperparameters for the Random Forest model on KDD dataset...
[*] Picked best hyperparms, now evaluating...
R2: 0.996388100577507
Best params: {'n_estimators': 50, 'min_samples_split': 4, 'min_samples_leaf': 5, 'max_samples': 10000, 'max_features': 0.5, 'max_depth': None}
[+] Evaluating the Random Forest model on KDD dataset...
Accuracy:  0.9954752927168089
========= Classification Report =========
              precision    recall  f1-score   support

           0      0.994     0.998     0.996     13457
           1      0.998     0.993     0.995     11738

    accuracy                          0.995     25195
   macro avg      0.996     0.995     0.995     25195
weighted avg      0.995     0.995     0.995     25195

========= Confusion Matrix =========
[[13429    28]
 [   86 11652]]
[+] Cross validating the Random Forest model...
{'fit_time': array([1.39534044, 2.35042405, 2.7372086 , 1.80262041, AccuraAccuracy for the fAccuracy for the fold no. 6 on the test set: 0.9962689529252997
Accuracy for the fold no. 7 on the test set: 0.9967452568071763Accuracy for the fold no. 8 on the test set: 0.9970627927284273Accuracy for the fold no. 9 on the test set: 0.9968246407874891



ipython RandomForest.py MIRAI
[+] Training the Random Forest model on MIRAI dataset...
[+] Picking the best hyperparameters for the Random Forest model on MIRAI dataset...
[*] Picked best hyperparms, now evaluating...
R2: 0.9993714285714286
Best params: {'n_estimators': 50, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_samples': 10000, 'max_features': 0.5, 'max_depth': 10}
[+] Evaluating the Random Forest model on MIRAI dataset...
Accuracy:  0.9990222222222223
========= Classification Report =========
              precision    recall  f1-score   support

      Benign      0.998     1.000     0.999     11130
   Malicious      1.000     0.998     0.999     11370

    accuracy                          0.999     22500
   macro avg      0.999     0.999     0.999     22500
weighted avg      0.999     0.999     0.999     22500

========= Confusion Matrix =========
[[11126     4]
 [   18 11352]]
[+] Cross validating the Random Forest model...
{'fit_time': array([5.42546606, 7.4524827 , 5.53026748, 6.40108776, 5.03117561,
       4.98650146, 4.94127679, 5.42726755, 5.42320442, 5.16076684]), 'scor,Accuracy for the fold no. 6 on the test set: 0.9997333333333334Accuracy for the fold no. 7 on the test set: 0.9997333333333334Accuracy for the fold no. 8 on the test set: 0.9997333333333334Accuracy for the fold no. 9 on the test set: 0.9089333333333334



 ipython RandomForest.py UCI
[+] Training the Random Forest model on UCI dataset...
[+] Picking the best hyperparameters for the Random Forest model on UCI dataset...
[*] Picked best hyperparms, now evaluating...
R2: 0.9998950817170353
Best params: {'n_estimators': 30, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_samples': 10000, 'max_features': 'sqrt', 'max_depth': 10}    
[+] Evaluating the Random Forest model on UCI dataset...
Accuracy:  0.9998887232101128
========= Classification Report =========
              precision    recall  f1-score   support

         ack      1.000     1.000     1.000     14903
      benign      0.999     1.000     1.000     14860
        scan      1.000     1.000     1.000     15013
         syn      1.000     1.000     1.000     14960
         udp      1.000     1.000     1.000     15030
    udpplain      1.000     1.000     1.000     15100

    accuracy                          1.000     89866
   macro avg      1.000     1.000     1.000     89866
weighted avg      1.000     1.000     1.000     89866

========= Confusion Matrix =========
[[14903     0     0     0     0     0]
 [    0 14860     0     0     0     0]
 [    0     1 15012     0     0     0]
 [    0     4     0 14956     0     0]
 [    0     1     2     0 15027     0]
 [    0     2     0     0     0 15098]]
[+] Cross validating the Random Forest model...
{'fit_time': array([3.19257426, 3.47621751, 3.36378813, 3.24502516, 3.40400934,
       3.44620895, 4.78043818, 4.77842426, 3.22920275, 2.2340703 ]), 'score_time': array([1.02742624, 0.83275342, 0.89974999, 1.42111921, 0.52066684,
774662Accuracy for the fold no. 6 on the test set: 0.999766316140878Accuracy for the fold no. 7 on the test set: 0.999866466366216
Accuracy for the fold no. 8 on the test set: 0.999799699549324Accuracy for the fold no. 9 on the test set: 0.999933233183108

"""


