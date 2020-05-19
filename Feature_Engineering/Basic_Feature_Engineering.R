####################################################
## Basic Feature Engineering Function (on inputs) ##
####################################################

## X: data.frame of inputs 

pairwise_sum <- function(X){
   
   ## Sanity check
   if(!is.data.frame(X)){
      X <- as.data.frame(X)
   }
   
   ## Here we compute the pairwise sum of all predictors
   data_out <- combn(ncol(X), 2, function(y) rowSums(X[, y]))
   data_out <- data.frame(data_out)
   
   ## Here we create the labels
   names(data_out) <- combn(names(X), 2, function(y) paste0(y[1], '_plus_', y[2]))
   
   return(data_out)
}


pairwise_diff <- function(X){
   
   ## Sanity check
   if(!is.data.frame(X)){
      X <- as.data.frame(X)
   }
   
   ## Here we compute the pairwise difference of all predictors
   data_out <- combn(ncol(X), 2, function(y) X[, y[1]] - X[, y[2]])
   data_out <- data.frame(data_out)
   
   ## Here we create the labels
   names(data_out) <- combn(names(X), 2, function(y) paste0(y[1], '_minus_', y[2]))
   
   return(data_out)
}


pairwise_prod <- function(X){
   
   ## Sanity check
   if(!is.data.frame(X)){
      X <- as.data.frame(X)
   }
   
   ## Here we compute the pairwise difference of all predictors
   data_out <- combn(ncol(X), 2, function(y) X[, y[1]] * X[, y[2]])
   data_out <- data.frame(data_out)
   
   ## Here we create the labels
   names(data_out) <- combn(names(X), 2, function(y) paste0(y[1], '_times_', y[2]))
   
   return(data_out)
}


inverse_features <- function(X){
   
   ## Sanity check
   if(!is.data.frame(X)){
      X <- as.data.frame(X)
   }
   
   ## Here we select columns with no zeroes
   data_out <- X[, !as.logical(apply(X, 2, function(x) any(x == 0)))]
   
   ## Here we compute the inverses
   data_out_1 <- data.frame(apply(data_out, 2, function(y) 1 / y))
   
   ## Here we add labels
   names(data_out_1) <- combn(names(data_out), 1, function(y) paste0(y, '_inverse'))
   
   return(data_out_1)
}


squared_features <- function(X){
   
   ## Sanity check
   if(!is.data.frame(X)){
      X <- as.data.frame(X)
   }
   
   ## Here we square the predictors
   data_out <- X^2
   
   ## Here we add the labels
   names(data_out) <- paste0(names(X), '_squared')
   
   return(data_out)
}


cubed_features <- function(X){
   
   ## Sanity check
   if(!is.data.frame(X)){
      X <- as.data.frame(X)
   }
   
   ## Here we square the predictors
   data_out <- X^3
   
   ## Here we add the labels
   names(data_out) <- paste0(names(X), '_cubed')
   
   return(data_out)
}