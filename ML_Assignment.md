# Machine Learning
========================================================

## Process 

The process of predicting the classe variable is as follows :

a) Extract training data
b) Remove any column with an NA variable. This reduces the number of predictors to 53.
c) We sample 30% of the file into a training dataset.
d) We use the random forest method to create our model.
e) As the number of predictors is large, we preprocess the data using the PCA.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.0.3
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```r
all_data <- read.csv("pml-training.csv", na.strings = c("NA", ""))
NAs <- apply(all_data, 2, function(x) {
    sum(is.na(x))
})
all_data <- all_data[, which(NAs == 0)]
set.seed(12345)
in_train <- createDataPartition(y = all_data$classe, p = 0.3, list = FALSE)
training <- all_data[in_train, ]
testing <- all_data[-in_train, ]
removeIndex <- grep("timestamp|X|user_name|new_window", names(training))
training <- training[, -removeIndex]
rfFit <- train(training$classe ~ ., data = training, method = "rf", prox = TRUE, 
    preProcess = "pca")
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'e1071' was built under R version 3.0.3
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```r
rfFit
```

```
## Random Forest 
## 
## 5889 samples
##   53 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction, scaled, centered 
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 5889, 5889, 5889, 5889, 5889, 5889, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.9       0.9    0.007        0.009   
##   30    0.9       0.9    0.01         0.01    
##   50    0.9       0.9    0.01         0.01    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```


Accuracy in-sample is 91.8% which is rather high. 

The out of sample is measured by running the model against the validation data or the testing variable.


```r
rfPred <- predict(rfFit, testing)
error_rate <- sum(rfPred == testing$classe)/length(testing$classe)
error_rate
```

```
## [1] 0.9415
```


The error rate is also high at 94%. We will employ this model on the test data.


```r
test_data <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
test_pred <- predict(rfFit, test_data)
```


The test_pred variable will be used to create the 20 files to be submitted. This should obtain a score of about 90%+.

