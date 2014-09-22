
library(nnet)
?multinom

library(caret)
library(RCurl)
library(Metrics)

#####################################################################
# Load data from Hadley Wickham on Github 
urlfile <-'https://raw.githubusercontent.com/hadley/fueleconomy/master/data-raw/vehicles.csv'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
vehicles <- read.csv(textConnection(x))

# clean up the data and only use the first 24 columns
vehicles <- vehicles[names(vehicles)[1:24]]
vehicles <- data.frame(lapply(vehicles, as.character), stringsAsFactors=FALSE)
vehicles <- data.frame(lapply(vehicles, as.numeric))
vehicles[is.na(vehicles)] <- 0

# use cyclinder column as outcome and cast to factor
vehicles$cylinders <- as.factor(vehicles$cylinders)
table(vehicles$cylinders)
#####################################################################


# shuffle and split
set.seed(1234)
vehicles <- vehicles[sample(nrow(vehicles)),]
split <- floor(nrow(vehicles)/2)
vehiclesTrain <- vehicles[0:split,]
vehiclesTest <- vehicles[(split+1):nrow(vehicles),]


# see how multinom predicts cylinders
# set the maxit to a large number, enough so the neural net can converge to smallest error
cylModel <- multinom(cylinders~., data=vehiclesTrain, maxit=500, trace=T)

# Sort by most influential variables
topModels <- varImp(cylModel)
topModels$Variables <- row.names(topModels)
topModels <- topModels[order(-topModels$Overall),]

# class/probs (best option, second best option?)
preds1 <- predict(cylModel, type="probs", newdata=vehiclesTest)
preds2 <- predict(cylModel, type="class", newdata=vehiclesTest)

# resample for accuracy - the mean squared error and R-squared are calculated of forfactors, the overall agreement rate and Kappa
postResample(vehiclesTest$cylinders,preds2)[[1]]
preds

library(Metrics)
classificationError <- ce(as.numeric(vehiclesTest$cylinders), as.numeric(preds))

# repeat cross validate by iterating through all the data to give every variable a chance of being test and train portions
totalError <- c()
cv <- 10
maxiterations <- 500 # try it again with a lower value and notice the mean error
cvDivider <- floor(nrow(vehiclesTrain) / (cv+1))

for (cv in seq(1:cv)) {
        # assign chunk to data test
        dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
        dataTest <- vehiclesTrain[dataTestIndex,]
        # everything else to train
        dataTrain <- vehiclesTrain[-dataTestIndex,]
        
        cylModel <- multinom(cylinders~., data=dataTrain, maxit=maxiterations, trace=T) 
        
        pred <- predict(cylModel, dataTest)
        
        #  classification error
        err <- ce(as.numeric(dataTest$cylinders), as.numeric(pred))
        totalError <- c(totalError, err)
}
print(paste('Mean error of all repeated cross validations:',mean(totalError)))

