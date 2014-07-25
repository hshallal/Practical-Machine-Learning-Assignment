---
title: "Practical Machine Learning Assignment"
author: "Hassan Shallal"
date: "July 23, 2014"
output: html_document
---

### Contents:
* Loading the training data

* Cleaning the training data

* Splitting the cleaning data into training and validation sets

* Building and validating a tree classification model
  
* Removing features that may lead to over-fitting
  
* Building and validating a second tree classification model

------------

### Loading the training data:

```{r, options(cache = TRUE)}
library(caret)
set.seed(12345)
sessionInfo()  ## report environment
training1 <- read.csv("pml-training.csv", stringsAsFactors = TRUE, na.strings = c("NA", ""))
training1$classe <- factor(training1$classe)
```

------------

### Cleaning the training data:

* By looking at the summary of the training1 data, there seem to be a large number of features of mainly missing data. These features won't be helpful in the training process and are going to be eliminated. only one variable was detected to have near-zero variance and hence was eliminated.

```{r}
training2 <- training1[colSums(is.na(training1)) < 1]
nsv <- nearZeroVar(training2, freqCut = 95/5, saveMetrics=TRUE) 
not_nsv <- subset(nsv, nsv$nzv == FALSE) 
not_nsv_variables <- rownames(not_nsv) ## A vector with non-near zero variance features
training3 <- training2[not_nsv_variables] ## subset the original training set
```

------------

### Splitting the cleaning data into training and validation sets:

```{r}
inTrain = createDataPartition(training3$classe, p = 3/4)[[1]]
train = training3[ inTrain,]
validate = training3[-inTrain,]
round(prop.table(table(train$classe))*100, digits = 1)
round(prop.table(table(validate$classe))*100, digits = 1)
```

------------

### Building and validating a tree classification model:

A tree model is chosen for the classification task in hand. No preprocessing was found to be necessary. The tree model performs 25 iteration of resampling using bootstrapping. 

```{r}
C5.0Tree_train <- train(classe ~ ., data = train, method = "C5.0Tree", allowParallel = TRUE); C5.0Tree_train ## Accuracy = 1
C5.0Tree_pred <- predict(C5.0Tree_train, validate)
confusionMatrix(C5.0Tree_pred, validate$classe)
imp_variables <- varImp(C5.0Tree_train)
imp_variables
```

By examining the important variables, none of the measured features was used for building the tree model. Character variables which may be correlated with the classification will be removed from both the training and the validation dataset.

------------

### Removing features that may lead to over-fitting:

```{r}
training4 <- training3[-c(1,2,5,6)] ## Deleting index number, user_name, cvtd_timestamp, and num_window variables
inTrain = createDataPartition(training4$classe, p = 3/4)[[1]]
train = training4[ inTrain,]
validate = training4[-inTrain,]
round(prop.table(table(train$classe))*100, digits = 1)
round(prop.table(table(validate$classe))*100, digits = 1)
```

------------

### Building and validating a second tree classification model:

```{r}
C5.0Tree_train <- train(classe ~ ., data = train, method = "C5.0Tree", allowParallel = TRUE); C5.0Tree_train ## Accuracy = 0.957
C5.0Tree_pred <- predict(C5.0Tree_train, validate)
confusionMatrix(C5.0Tree_pred, validate$classe) ## Accuracy : 0.9753 
imp_variables <- varImp(C5.0Tree_train)
imp_variables
```

* The built model accurately predicted 19 out of 20 cases. It only misclassified one case of C class as E class.