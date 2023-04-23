\# Best Models For Each Dataset

Iot23: 
Model: Decision Tree 

Reasoning: SVM results are especially poor
in comparison to both other models. The RFC has a higher average
accuracy in the k-folds evaluation, however its average precision and
recall are the worst between all 3 models. The Decision Tree has the
best model accuracy, better average accuracy than the SVM, and better
average precision and recall than the RFC.

UCI: 
Model: Random Forest Classifier 

Reasoning: The SVM has the overall
best model accuracy, precision, and recall. It\'s average metrics are
outperformed by the RFC however. Both are sufficient models for this
dataset but the RFC may be capable of producing better results on
average. SVM also suffersfrom increased runtime

NSL-KDD: 
Model: Of the three chosen: Random Forest Classifier Generally:
Inconclusive

Reasoning: The Decision Tree provides the best overall model performance
while the Random Forest Classifier provides the best average
performance. However, none of the models are especially impressive nor
consistent when regarding this model. Further research needs to be done
to find a suitable model for this dataset.

Kitsune: 
Model: Support Vector Machine 

Reasoning: The Support Vector
Machine resulted in the best overall and average metrics across the
board. In addition, performance is incredibly efficient with the longest
time taken to fit the training data being \< 40 seconds.

Overall Best Model: Random Forest Classifier

Reasoning: While its overall
performance was typically outclassed by the Decision Tree, it performed
better on average. While results are not consistently desirable, they
are more likely to be sufficient than the other two models

Overall Worst Model: Support Vector Machine 

Reasoning: Kitsune dataset
aside, this model fairly consistently produced the worst metrics both
overall and on average. To make matters worse, performance was
significantly slower than the other two models, with some of the times
taken to fit the model to the training data taking up to 2 hours for
some datasets
