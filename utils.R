## Utilities files
## Load functions.

## Create a testing and train set
## Create a testing and train set
set.seed(2017)
samp <- sample(nrow(explore.df), 0.6 * nrow(explore.df))

## Setting up the working directory for all data file references.
# Ensure all data files for use are in the working directory.
workingDirectory <- "D://data//other-projects//R//titanic" # Place all data files here #
setwd(workingDirectory)
work_dir <- getwd()

## Function for exporting tables to CSVs. 
##Directory path other than work directory and extension needs to be defined 'x', 'y' is the table to be exported.
file_output <- function (x, y){
  path <- paste(work_dir, x, sep ="")
  write.table(y, file = path, sep = ",", row.names = FALSE)
}

## Create a matrix for correlations and significance
## Based on output from cor
# http://www.r-bloggers.com/more-on-exploring-correlations-in-r/
cor.prob <- function(X, dfr = nrow(X) - 2) {
  R <- cor(X)
  above <- row(R) < col(R)
  r2 <- R[above]^2
  Fstat <- r2 * dfr / (1 - r2)
  R[above] <- 1 - pf(Fstat, 1, dfr)
  R
}

## Create a matrix for correlations and significance
## Based on output from rcorr
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}


## The assessment stack as a single function
accuracyAssess <- function(cm){
  
  results <- list()
  
  ## Much clearer
  # http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
  
  ## Load in a crostable of frequecies for actual and predicted groups
  ## Predicted is the columns and actuals are the rows.
  ## Creating accuracy measures
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class 
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  results$accuracy = sum(diag) / n 

  precision = diag / colsums # fraction of correct predictions for a certain class
  recall = diag / rowsums # fraction of instances of a class that were correctly predicted
  f1 = 2 * precision * recall / (precision + recall) # harmonic mean (or a weighted average) of precision and recall
  results$PrecRecf1 <- data.frame(precision, recall, f1) 
  
  macroPrecision = mean(precision)
  macroRecall = mean(recall)
  macroF1 = mean(f1)
  results$macro <- data.frame(macroPrecision, macroRecall, macroF1)
  
  results$oneVsAll = lapply(1 : nc,
                    function(i){
                      v = c(cm[i,i],
                            rowsums[i] - cm[i,i],
                            colsums[i] - cm[i,i],
                            n-rowsums[i] - colsums[i] + cm[i,i]);
                      return(matrix(v, nrow = 2, byrow = T))})
  
  s = matrix(0, nrow = 2, ncol = 2)
  for(i in 1 : nc){results$s = s + results$oneVsAll[[i]]}

  results$avgAccuracy = sum(diag(s)) / sum(s)

  results$micro_prf = (diag(s) / apply(s,1, sum))[1];

  # Evaluation on Highly Imbalanced Datasets
  # Because this is what we actually have here
  
  results$mcIndex = which(rowsums==max(rowsums))[1] # majority-class index
  results$mcAccuracy = as.numeric(p[results$mcIndex]) 
  mcRecall = 0*p;  mcRecall[results$mcIndex] = 1
  mcPrecision = 0*p; mcPrecision[results$mcIndex] = p[results$mcIndex]
  mcF1 = 0*p; mcF1[results$mcIndex] = 2 * mcPrecision[results$mcIndex] / (mcPrecision[results$mcIndex] + 1)
  
  # Expected accuracy for majority class.
  results$mc <- data.frame(mcRecall, mcPrecision, mcF1) 
  
  ## Random guess
  (n / nc) * matrix(rep(p, nc), nc, nc, byrow=F)
  results$rgAccuracy = 1 / nc
  rgPrecision = p
  rgRecall = 0*p + 1 / nc
  rgF1 = 2 * p / (nc * p + 1)
  
  results$rg <- data.frame(rgPrecision, rgRecall, rgF1)
  
  ## Random weighted guesses
  n * p %*% t(p)
  results$rwgAccurcy = sum(p^2)
  rwgPrecision = p
  rwgRecall = p
  rwgF1 = p
  
  results$rwg <- data.frame(rwgPrecision, rwgRecall, rwgF1)
  
  ## Kappa
  expAccuracy = sum(p*q)
  results$kappa = (results$accuracy - expAccuracy) / (1 - expAccuracy)

  return(results)
   
}

## Return the kappa
kappaOnline <- function(x, y){
  cm <- table(x, y)
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class 
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  accuracy = sum(diag) / n 
  ## Kappa
  expAccuracy = sum(p*q)
  (accuracy - expAccuracy) / (1 - expAccuracy)
}
