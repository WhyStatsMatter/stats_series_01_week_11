# Snippet 1: Bias-Variance Tradeoff and Model Complexity
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Snippet 2: Cross-Validation Techniques
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv_scores = cross_val_score(model, X, y, cv=5) # K-Fold Cross-Validation
stratified_cv = StratifiedKFold(n_splits=5)
stratified_scores = cross_val_score(model, X, y, cv=stratified_cv) # Stratified K-Fold

# Snippet 3: Regularization Techniques
from sklearn.linear_model import Lasso, Ridge, ElasticNet

lasso = Lasso().fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)
elastic_net = ElasticNet().fit(X_train, y_train)

# Snippet 4: Hyperparameter Tuning and Model Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
final_accuracy = accuracy_score(y_test, grid_search.predict(X_test))
