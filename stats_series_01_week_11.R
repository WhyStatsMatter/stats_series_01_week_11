# Snippet 1: Bias-Variance Tradeoff and Model Complexity
trainIndex <- sample(1:nrow(df), 0.7*nrow(df))
trainData <- df[trainIndex,]
testData <- df[-trainIndex,]

# Snippet 2: Cross-Validation Techniques
library(caret)
cv <- trainControl(method="cv", number=5)
model_cv <- train(y ~ ., data=df, trControl=cv)

# Snippet 3: Regularization Techniques
library(glmnet)
lasso <- glmnet(x=X_train, y=y_train, alpha=1)
ridge <- glmnet(x=X_train, y=y_train, alpha=0)
elastic_net <- glmnet(x=X_train, y=y_train, alpha=0.5)

# Snippet 4: Hyperparameter Tuning and Model Evaluation
grid <- expand.grid(C=c(0.1, 1, 10), kernel=c('linear', 'rbf'))
tune_result <- train(y ~ ., data=trainData, method='svmRadial', tuneGrid=grid)
final_accuracy <- postResample(predict(tune_result, testData), testData$y)
