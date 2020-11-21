# sklearn 

```py
import sklearn
sklearn.show_versions()

from sklearn.datasets import load_boston

from sklearn.emsemble import RandomForestClassifier
from sklearn.emsemble import RandomForestRegressor
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import OneHotEncoder
pd.get_dummies()
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
imputer.fit_transform()
imputer.transform()

np.random.seed(42)

df.pop()
df.drop('column', axis=1)
df.isna().sum()
df.dropna(inplace=True)

model.get_params()
model.fit()
model.predict()
model.predict_proba()
model.score()

df.shape

# 5  improve a model 
## try different hyperparameters
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(i).fit(x_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) *100:.2f}%")
    print("")
    
# 6 save a module and load it
import pickle

pickle.dump(clf, open("random_forest_model_1.pk1", "wb"))

loaded_model = pickle.load(open("random_forest_model_1.pk1", "rb"))
loaded_model.score(x_test, y_test)
```
