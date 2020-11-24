## conda cheatsheet
https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf

## setting up the conda environment
COMPLETE THIS SECTION.

## Tip to download a library using conda in a Jupyter Notebook.
```py
import sys
#!conda install --yes --prefix {sys.prefix} <module>
```
# sklearn - tabular data machine learning

```py
import sklearn
sklearn.show_versions()

from sklearn.datasets import load_boston

from sklearn.emsemble import RandomForestClassifier
from sklearn.emsemble import RandomForestRegressor
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

# Metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
# classification metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import absolute_mean_error
from sklearn.model_selection import cross_val_score
## binary metrics
from sklearn.metrics import roc_curve # fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
from sklearn.metrics import roc_auc_score # gives area under the roc_curve.
from sklearn.metrics import confusion_matrix
# note that from sklearn.metrics import plot_confusion_matrix is not working at the moment. Two engineered solutions are provided below in graphs section.
from sklearn.metrics import classification_report

# regression metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# preprocessing
from sklearn.preprocessing import OneHotEncoder
pd.get_dummies()
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
imputer.fit_transform()
imputer.transform()

np.random.seed(42)

# create data
df.pop()
df.drop('column', axis=1)
df.isna().sum()
df.dropna(inplace=True)

# model attributes
model.get_params()
model.fit()
model.predict()
model.predict_proba()
model.score()

df.shape

## Graphs

#1. ROC curve
# import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr):
    '''
    Plots a roc curve given the false positive rate (fpr) and the true positive rate (tpr) of model.
    '''
    # plot the roc curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    # plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
    #custimise the plot 
    plt.xlabel("False Positive Rate (fpr)")
    plt.ylabel("True Positive Rate (tpr)")
    plt.title("Receiver Operating Characteristics (ROC) Curve")
    plt.legend()
    plt.show()

#2a. Confusion Matric
# visualise confustion matrix with pd.crosstab()
pd.crosstab(y_test,
           y_preds,
           rownames=['Actual Labels'],
           colnames=['Predicted Labels'])
           
#2b. Confusion Matric
# Or make the confustion matrix more visual with seaborn heatmap
import seaborn as sns

# Set the font scale
sns.set(font_scale=1.5)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

def plot_conf_mat(conf_mat):
    '''
    Plots a confustion matrix using seaborns heatmap()
    '''
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat, 
                    annot=True,
                    cbar=False) # annotate the boxes with conf_mat info
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

plot_conf_mat(conf_mat)

## Tuning Hyperparameters: Function to return evaluation metrics.

def evaluate_preds(y_true, y_preds):
    '''
    Perfoms evaluation comparison on y_true labels vs. y_preds labels on a classification.
    '''
    accuracy = accuracy(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2), 
                   "precision": round(precision, 2),
                  "recall": round(recall, 2),
                  "f1": round(f1, 2)
                  }
    print(f'Acc: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}')
    print(f'Recall: {recall * 100:.2f}')
    print(f'F1: {f1 * 100:.2f}')
    
    return metric_dict



# improve a model 
## try different hyperparameters
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(i).fit(x_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) *100:.2f}%")
    print("")
    
# save a module and load it
import pickle

pickle.dump(clf, open("random_forest_model_1.pk1", "wb"))

loaded_model = pickle.load(open("random_forest_model_1.pk1", "rb"))
loaded_model.score(x_test, y_test) 
```
