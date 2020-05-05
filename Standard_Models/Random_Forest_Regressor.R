#############################
## Random Forest Regressor ##
#############################

Random_Forest_Regressor <- function(X, Y){
   
   ## Checking for randomforest package
   if (!require(randomForest, character.only = T, quietly = T)) {
      install.packages(randomForest)
      library(randomForest, character.only = T)
   }
   
   ## This function assumes that X is the data.frame/matrix of
   ## input and Y is the target varible
   rf_md <- randomForest(Y ~ X)
   
   
   
   
}