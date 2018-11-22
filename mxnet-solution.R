## mxnet Solution
library(mxnet)

## Initial load of the data
## Observations per row format.
## Loading the pre formated and transformed data. Scaling and dummy variables have already been applied
mxnet.ma <- explore.ma[,-1]

## Will need to manually split the frame.
train.mxn.x = data.matrix(mxnet.ma[-samp, 1:25]) # Factors
train.mxn.y = as.numeric(unlist(mxnet.ma[-samp, 27]))-1 # Response
test.mxn.x = data.matrix(mxnet.ma[samp, 1:25]) # Factors
test.mxn.y = as.numeric(unlist(mxnet.ma[samp, 27]))-1 # Response

mx.set.seed(0)

## Running a simple NN model.
model.mxn <- mx.mlp(train.mxn.x, train.mxn.y, 
                    hidden_node = 10, 
                    out_node = 3, 
                    out_activation = "softmax", 
                    num.round = 50, 
                    array.batch.size = 20, 
                    learning.rate = 0.05, 
                    momentum = 0.9,
                    eval.metric = mx.metric.accuracy)

## Visualise the model constructed
graph.viz(model.mxn$symbol)

## Checking the prediction.
preds = predict(model.mxn, test.mxn.x)
pred.label = max.col(t(preds))-1
table(pred.label, test.mxn.y)

## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = test.mxn.y, Predicted = pred.label ))
cm

accuracyAssess.mxns <- accuracyAssess(cm)

## Explicit feed forward network

## This is defining the nn function,
## The refence to "data" is the input
small_net = function() {
  # incoming data
  data <- mx.symbol.Variable("data")
  
  # hidden layer
  fc1 <- mx.symbol.FullyConnected(data, num_hidden = 10)
  # activation
  act <- mx.symbol.Activation(fc1, act_type = 'tanh')
  
  # output layer
  fc2 <- mx.symbol.FullyConnected(act, num_hidden = 4)
  
  # loss function
  net <- mx.symbol.SoftmaxOutput(data = fc2)
  return(net)
}

## Running the model through training
net <- small_net()
model.mxnff <- mx.model.FeedForward.create(net, X = train.mxn.x, y = train.mxn.y,
                                     num.round = 50, 
                                     array.batch.size = 20, 
                                     learning.rate = 0.05, 
                                     momentum = 0.9, 
                                     eval.metric = mx.metric.accuracy)

## Visualise the model constructed
graph.viz(model.mxnff$symbol)

## Checking the prediction.
predsff = predict(model.mxnff, test.mxn.x)
predff.label = max.col(t(predsff))-1
table(predff.label, test.mxn.y)

## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = test.mxn.y, Predicted = predff.label ))
cm

accuracyAssess.mxnf <- accuracyAssess(cm)

