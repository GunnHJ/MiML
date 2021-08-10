###################################################################################################
###################################################################################################
# Code used for Multiple Imputation Machine Learning (MiML) paper
###################################################################################################
###################################################################################################

# calculate start time to determine how long analyses take to run
start.time.total <- Sys.time()

################################################################
# Call up libraries
################################################################

# install libraries (only needs to be done once unless there's an update)
#install.packages('glmnet')
#install.packages('splitstackshape')
#install.packages('psych')
#install.packages('mice')

# call up libraries
library(glmnet) # used for LASSOs
library(splitstackshape) # used for getanID
library(psych) # used for describe function
library(mice) # use when analyzing FMI data

############################################################################
############################################################################
# Prepare data for all 3 imputation LASSO methods and listwise deletion
############################################################################
############################################################################

################################################################
# Import Imputed Data from Blimp 
################################################################

# Import imputed data and psr values from Blimp
WRKDIR <- "C:/Users/.../" # Modify to your file path
fullData=read.table(paste0(WRKDIR,"imps_stacked.dat"))
psr=read.csv(paste0(WRKDIR,"PSRs_with_Labels.csv"), header = TRUE)

# Assign names to variables (need to use .imp to use mice package)
colnames(fullData)<-c(".imp","depression_raw","age","audit_c","count_lf","ptsd","anxietyr","sadr","intfsocr","energr",
                      "anxiety_raw","srMentH","srPhysH","srLivSit","srDrugFree","srSocNet","srSexRx","ssHelp",
                      "ssEmoHelp","ssNewFri","smAppFreq","datingAppFreq","losAng","RaceEthnicity","female","cis",
                      "heterosexual","employed","poverty","insured","healthProvider","medicalUtil_re","erCare_re",
                      "sapro_lf","hivpp_lf","homeless_lf","incarceration_lf","ipv_lf","sexex_lf","suicide_lf",
                      "hospitalmh_lf","sexab_lf","tmx5old_lf","tmrob_lf","tmotinj_lf","tmotmdr_lf","xdrug_r","smoke_lf")

# Note on imp variable:
# imp variable designates imputed data
# imp = 0 is original data set with missing values
# imp = 1-m designate the m imputed data sets (m = 50 for our example)

# Recode missing values from 999 to NA
fullData[fullData==999] <- NA

################################################################
# Create dummy codes for categorical variables with more than 2 categories
################################################################

# Designate race/ethnicity variable as categorical
fullData$RaceEthnicity <- factor(fullData$RaceEthnicity)

# Create dummy codes for all categories of race/ethnicity variable
fullData[,(ncol(fullData)+1):(ncol(fullData)+4)] = model.matrix(~RaceEthnicity+0,data=fullData)
names(fullData)[(ncol(fullData)-3):(ncol(fullData))]<-c("Black","Latinx","White","Other")

# Remove categorical variable that created the dummy variables
fullData <- subset(fullData, select=-c(RaceEthnicity))

################################################################
# Descriptive Statistics, Table 1
################################################################

# Select original data set (contains missing values) and remove ".imp" column
original <- fullData[fullData$.imp==0,-1]

# Calculate means and standard deviations of all variables, format M (SD)
datmeans <- round(describe(original)$mean,2)
datsd <- round(describe(original)$sd,2)
datsd <- paste0("(",format(unlist(datsd)),")")
datmeansd <- as.data.frame(paste(format(datmeans,digits=2),datsd))

# Calculate number of missing values per variable as well as percentage, format n (%)
datmiss <- sapply(original,function(x) sum(is.na(x)))
percmiss <- round(datmiss/nrow(original)*100,2)
percmiss <- paste0("(",format(unlist(percmiss),trim=TRUE),")")
datpercmiss <- as.data.frame(paste(datmiss,percmiss))

# Create Table
Table1 <- cbind(names(original),datmeansd,datpercmiss)
names(Table1) <- c("Variable", "M (SD)", "Number Missing (%)")

# Export table as csv file
#write.csv(Table1, paste0(WRKDIR,"Table1.csv"))

################################################################
# Create Training and Test Data Sets
################################################################

# Moving depression_raw, the outcome, to end of data set (Makes it easier to fit LASSOs)
fullData <- fullData[,c(1,3:51,2)]

# Create original data set with an id var
originalID <- getanID(fullData, id.vars = ".imp")
originalID <- originalID[which(originalID$.imp==0),]

# Select id variable for participants in complete case analysis and not
idNoMiss <- originalID[complete.cases(originalID),52]
idMiss <- originalID[rowSums(is.na(originalID)) > 0,52]

# Shuffle rows to create random sample for training and test data sets
#set.seed(73346)
set.seed(69437)
idNoMiss<-idNoMiss[sample(nrow(idNoMiss)),]
set.seed(90259)
idMiss<-idMiss[sample(nrow(idMiss)),]

# Create 75/25 split for training/test data sets separately for participants in complete case analysis
# and for participants not in that analysis
cutoff <- .75*nrow(idNoMiss)
idNoMiss_Train <- idNoMiss[1:cutoff,1]
idNoMiss_Test <- idNoMiss[(cutoff+1):(nrow(idNoMiss)+1),1]

cutoff <- .75*nrow(idMiss)
idMiss_Train <- idMiss[1:cutoff,1]
idMiss_Test <- idMiss[(cutoff+1):(nrow(idMiss)+1),1]

# Combine the IDs for participants with complete data and with missing data
id_Train <- rbind(idNoMiss_Train,idMiss_Train)
id_Test <- rbind(idNoMiss_Test,idMiss_Test)

# Recreate data set (original and imputed data sets) with id var
fullDataID <- getanID(fullData, id.vars = ".imp")

# Create training data set, selecting participants from randomized id variable
TrainData <- fullDataID[fullDataID$.id %in% id_Train$.id,-52]

# Create test data set
TestData <- fullDataID[fullDataID$.id %in% id_Test$.id,-52]

# Optional: Remove unnecessary data sets and values to clean up RStudio workspace (these data sets are not used in future code)
rm(fullData,originalID,idNoMiss,idNoMiss_Test,idNoMiss_Train,idMiss,idMiss_Test,idMiss_Train,id_Test,id_Train,cutoff)
rm(datmeansd,datpercmiss,datmeans,datmiss,datsd,percmiss,original)

###############################################################
# Create empty matrices to save output for the 4 approaches
###############################################################

# 4 approaches = Listwise, Separate, Stacked, MI-LASSO

# Define parameters (number of imputations, number of predictors, number of folds, sample size)
m = 50
P = ncol(TrainData) - 2 # subtract outcome and imputation columns
k = 10
N = nrow(fullDataID[fullDataID$.imp==0,-1])
Ntrain = nrow(TrainData[TrainData$.imp==0,-1])
Ntest = nrow(TestData[TestData$.imp==0,-1])

# Save regression coefficients (including intercept) in final model for all approaches
# Listwise deletion has 1 option
# Separate approach has 5 different options
# Stacked approach has 3 different options
# MI-LASSO has 1 option
# 1+5+3+1 = 10 columns
Coefficients = matrix(NA,P+1,10)

# Save fit measures (MSE and R-squared) and optimal lambda for all approaches
Fit = matrix(NA,5,10)
#colnames(Fit)<-c("Listwise","SepAvg","SepMed","SepMin","SepMax","SepInd","StackedW","StackedW2",
#                 "StackedNW","MI-LASSO")
#rownames(Fit)<-c("R2 training","MSE training","R2 test","MSE test","lambda")


############################################################################
############################################################################
# Listwise Deletion
############################################################################
############################################################################

##############################################
# Prepare data
##############################################

# Select original data set (contains missing values) and remove "imp" column
originalTrain <- TrainData[TrainData$.imp==0,-1]
originalTest <- TestData[TestData$.imp==0,-1]

# Create data set with only participants who have no missing data on any variables
listwiseTrain <- originalTrain[complete.cases(originalTrain),]
listwiseTest <- originalTest[complete.cases(originalTest),]

# Calculating sample sizes
Nlist <- nrow(listwiseTrain)+nrow(listwiseTest)
nrow(listwiseTrain)
nrow(listwiseTest)

##############################################
# Cross-validation
##############################################

# 10-fold cross-validation to determine optimal lambda via MSE
# x = matrix of predictors
# y = vector of outcome scores
# alpha = 1: specifies the LASSO, 0: specifies ridge regression, any # in between: specifies elastic net
# standardize: standardizes the predictors, but returned coefficients are on original scale
# intercept: estimate the intercept (don't constrain it to be 0)
# type.measure: measure of fit/error to determine optimal lambda
# nfolds: specify number of folds (k=10 for this example)
set.seed(1989)
cv.listwise <- cv.glmnet(x=data.matrix(listwiseTrain[,1:P]),y=listwiseTrain$depression_raw, 
                         alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)

# Save value of lambda that gives minimum cvm (mean cross-validated error)
lambdaListwise <- cv.listwise$lambda.min
Fit[5,1] <- lambdaListwise

# Checking what cv.glmnet is doing
cv.listwise$lambda
cv.listwise$cvm
check = glmnet(x=data.matrix(listwiseTrain[,1:P]),y=listwiseTrain$depression_raw,
                        alpha=1,standardize=TRUE,intercept=TRUE,lambda=4.464565169)
as.array(check$beta)
check = glmnet(x=data.matrix(listwiseTrain[,1:P]),y=listwiseTrain$depression_raw,
               alpha=1,standardize=TRUE,intercept=TRUE,lambda=4.067945504)
as.array(check$beta)
check = glmnet(x=data.matrix(listwiseTrain[,1:P]),y=listwiseTrain$depression_raw,
               alpha=1,standardize=TRUE,intercept=TRUE,lambda=0.003793780)
as.array(check$beta)

##############################################
# Training Model
##############################################

# LASSO on full training data using cross-validated lambda
train.listwise = glmnet(x=data.matrix(listwiseTrain[,1:P]),y=listwiseTrain$depression_raw,
                        alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdaListwise)

# Save regression coefficients (beta) including intercept (a0)
Coefficients[2:nrow(Coefficients),1] <- as.array(train.listwise$beta)
Coefficients[1,1] <- unname(train.listwise$a0)

# Calculate predicted scores for each participant based on the model (link gives linear predictors, default)
predict_listwise_train <- predict(train.listwise,newx=data.matrix(listwiseTrain[,1:P]),type="link")

# Calculate MSE, mean of Equation 1 in paper
Fit[2,1] <- mean((listwiseTrain$depression_raw - predict_listwise_train)^2)

# Calculate R-squared in training set
Fit[1,1] <- train.listwise$dev.ratio

##############################################
# Test Model
##############################################

# Calculate predicted scores for each participant in test set based on the training model
predict_listwise_test <- predict(train.listwise,newx=data.matrix(listwiseTest[,1:P]),type="link")
# Calculate MSE, mean of Equation 1 in paper
Fit[4,1] <- mean((listwiseTest$depression_raw - predict_listwise_test)^2)
# Notice that mean of residuals is not 0 in test set, thus r^2 must be calculated with RSS, not predicted sum of squares
mean(listwiseTest$depression_raw - predict_listwise_test)

# Calculate residual sum of squares in test set
rss_listwise_test <- sum((listwiseTest$depression_raw - predict_listwise_test)^2)
# Calculate total sum of squares in test set
tss_listwise_test <- sum((listwiseTest$depression_raw - mean(listwiseTest$depression_raw))^2)
# Calculate R-squared in test set
Fit[3,1] <- 1 - rss_listwise_test/tss_listwise_test


############################################################################
############################################################################
# Separate imputed data sets
############################################################################
############################################################################

##############################################
# Prepare data
##############################################

# Remove original data set and only keep imputed data sets
imputedTrain <- TrainData[TrainData$.imp!=0,]
imputedTest <- TestData[TestData$.imp!=0,]

##############################################
# Cross-validation
##############################################

# Save all m optimal lambda values
lambdaSeparate = matrix(NA,m)

# calculate start time to determine how long cross-validation takes to run
start.time.sep <- Sys.time()

# Run k-fold cross validation on each imputed data set independently and save optimal lambda value
for(d in 1:m){
  # Create one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  
  # 10-fold Cross-validation to find lambda (for that imputed data set)
  set.seed(17463)
  cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                           alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
  
  # save value of lambda that gives minimum cvm (mean cross-validated error) for that imputed data set
  lambdaSeparate[d] <- cv.separate$lambda.min
}

# calculate end time to determine how long cross-validation takes to run
end.time.sep <- Sys.time()
time.taken.sep <- end.time.sep - start.time.sep

lambdaSepAvg = mean(lambdaSeparate)
lambdaSepMed = median(lambdaSeparate)
lambdaSepMin = min(lambdaSeparate)
lambdaSepMax = max(lambdaSeparate)
Fit[5,2] <- lambdaSepAvg
Fit[5,3] <- lambdaSepMed
Fit[5,4] <- lambdaSepMin
Fit[5,5] <- lambdaSepMax

# Distribution of lambda values
#hist(lambdaSeparate, xlab="Lamba")

# Checking within-imputation lambda value
oneImputed <- imputedTrain[which(imputedTrain$.imp==1),-1]
set.seed(74592)
cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                         alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
cv.separate$lambda.min

# Allowing lambda values to have within- and between-imputation variance
lambdaSeparateWB = matrix(NA,m)
set.seed(17463)
for(d in 1:m){
  # Create one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  
  # 10-fold Cross-validation to find lambda (for that imputed data set)
  cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                           alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
  
  # save value of lambda that gives minimum cvm (mean cross-validated error) for that imputed data set
  lambdaSeparateWB[d] <- cv.separate$lambda.min
}
mean(lambdaSeparateWB)
median(lambdaSeparateWB)
min(lambdaSeparateWB)
max(lambdaSeparateWB)
Fit[5,2:5]

##############################################
# Training Data
##############################################

# Save all m sets of coefficients for the different lambda values
coefSeparAvg = matrix(NA,P+1,m)
coefSeparMed = matrix(NA,P+1,m)
coefSeparMin = matrix(NA,P+1,m)
coefSeparMax = matrix(NA,P+1,m)
coefSeparInd = matrix(NA,P+1,m)

# Fit LASSO in training data for each imputed data set using cross-validated lambda
for(d in 1:m){
  # Create one imputed data set (remove column that contains imputation number)
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]

  # LASSO using averaged lambda value
  train.separate.avg = glmnet(x=data.matrix(oneImputed[,1:P]),y=oneImputed$depression_raw,
                          alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdaSepAvg)
  
  # Save regression coefficients including intercept
  coefSeparAvg[2:nrow(coefSeparAvg),d] <- as.array(train.separate.avg$beta)
  coefSeparAvg[1,d] <- unname(train.separate.avg$a0)
  
  # LASSO using median lambda value
  train.separate.med = glmnet(x=data.matrix(oneImputed[,1:P]),y=oneImputed$depression_raw,
                          alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdaSepMed)
  # Save regression coefficients including intercept
  coefSeparMed[2:nrow(coefSeparMed),d] <- as.array(train.separate.med$beta)
  coefSeparMed[1,d] <- unname(train.separate.med$a0)
  
  # LASSO using minimum lambda value
  train.separate.min = glmnet(x=data.matrix(oneImputed[,1:P]),y=oneImputed$depression_raw,
                              alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdaSepMin)
  # Save regression coefficients including intercept
  coefSeparMin[2:nrow(coefSeparMin),d] <- as.array(train.separate.min$beta)
  coefSeparMin[1,d] <- unname(train.separate.min$a0)
  
  # LASSO using maximum lambda value
  train.separate.max = glmnet(x=data.matrix(oneImputed[,1:P]),y=oneImputed$depression_raw,
                              alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdaSepMax)
  # Save regression coefficients including intercept
  coefSeparMax[2:nrow(coefSeparMax),d] <- as.array(train.separate.max$beta)
  coefSeparMax[1,d] <- unname(train.separate.max$a0)
  
  # LASSO using individualized lambda value
  train.separate.ind = glmnet(x=data.matrix(oneImputed[,1:P]),y=oneImputed$depression_raw,
                              alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdaSeparate[d])
  # Save regression coefficients including intercept
  coefSeparInd[2:nrow(coefSeparInd),d] <- as.array(train.separate.ind$beta)
  coefSeparInd[1,d] <- unname(train.separate.ind$a0)
}

####### Calculate Inclusion Frequencies and averaged coefficients of Final Lasso Model ######## 

# Define threshold (variable must be selected in this % of imputed data sets or more)
threshold = 50

# Calculate inclusion frequency (IF), percentage of data sets variable was selected
# If IF less than threshold, set average coefficient to 0 for that variable
# If IF greater than threshold, average the regression coefficients across imputed data sets
InclFreq = matrix(NA,P+1,5)

for(i in 1:nrow(coefSeparAvg)){
  InclFreq[i,1] <- (m - sum(coefSeparAvg[i,1:m] == 0))/m*100
  InclFreq[i,2] <- (m - sum(coefSeparMed[i,1:m] == 0))/m*100
  InclFreq[i,3] <- (m - sum(coefSeparMin[i,1:m] == 0))/m*100
  InclFreq[i,4] <- (m - sum(coefSeparMax[i,1:m] == 0))/m*100
  InclFreq[i,5] <- (m - sum(coefSeparInd[i,1:m] == 0))/m*100
  
  if(InclFreq[i,1] < threshold){
    Coefficients[i,2] = 0
  } else {
    Coefficients[i,2] = mean(coefSeparAvg[i,1:m])
  }
  if(InclFreq[i,2] < threshold){
    Coefficients[i,3] = 0
  } else {
    Coefficients[i,3] = mean(coefSeparMed[i,1:m])
  }
  if(InclFreq[i,3] < threshold){
    Coefficients[i,4] = 0
  } else {
    Coefficients[i,4] = mean(coefSeparMin[i,1:m])
  }
  if(InclFreq[i,4] < threshold){
    Coefficients[i,5] = 0
  } else {
    Coefficients[i,5] = mean(coefSeparMax[i,1:m])
  }
  if(InclFreq[i,5] < threshold){
    Coefficients[i,6] = 0
  } else {
    Coefficients[i,6] = mean(coefSeparInd[i,1:m])
  }
}

# Save variable names, modify intercept name
VarNames <- rownames(coef(train.separate.avg))
VarNames[1] <- "Intercept"

# Apply variable names to rows of Inclusion Frequency matrix
colnames(InclFreq)<-c("Mean","Median","Min","Max","Ind")
rownames(InclFreq) <- VarNames
InclFreq <- as.data.frame(InclFreq)

###### Comparing the 5 separate approaches 
# Order in ascending order
#InclFreq <- InclFreq[order(InclFreq[,4], InclFreq[,3]),]

# Did variable meet IF threshold?
Keep <- ifelse(InclFreq >= 50,1,0)
# Order in ascending order
Keep <- Keep[order(Keep[,4], Keep[,1], Keep[,3]),]

# Number of variables selected by each lambda option (subtract intercept)
sum(Keep[2:49,1])
sum(Keep[2:49,2])
sum(Keep[2:49,3])
sum(Keep[2:49,4])
sum(Keep[2:49,5])

# For mean lambda, how many variables selected in 1, half, all imputed data sets? (subtract intercept)
sum(InclFreq$Mean > 0) - 1
sum(InclFreq$Mean >= 50) - 1
sum(InclFreq$Mean == 100) - 1

####### Calculate Fit Statistics ########

# Calculate predicted scores for each participant in training set based on the averaged coefficients
x = data.matrix(imputedTrain[,2:50]) # select all columns of predictors
y = as.numeric(imputedTrain$depression_raw) # select outcome
predTrainSep = matrix(NA,nrow(imputedTrain),6) # create empty matrix to store predicted values
for(j in 2:6){
  b0 = Coefficients[1,j]
  b = Coefficients[2:50,j]
  predTrainSep[1:nrow(imputedTrain),j] = x%*%b+b0 # calculate predicted y scores based on model
}

# Adding column to designate imputed data set
predTrainSep[,1] <- imputedTrain$.imp

# Calculate fit statistics within each imputed data set using the final model
rsqSepTrain = matrix(NA,m,5)
mseSepTrain = matrix(NA,m,5)
for(d in 1:m){
  # Create observed variable for one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  y <- as.numeric(oneImputed$depression_raw)
  
  # Create predicted scores for one imputed data set
  onePredict <- predTrainSep[which(predTrainSep[,1]==d),-1]
  
  # Calculate R-squared in training set for all 5 lambda values
  # Calculate MSE, mean of Equation 1 in paper
  for(l in 1:5){
    rsqSepTrain[d,l] = 1 - sum((y - onePredict[,l])^2)/sum((y - mean(y))^2)
    mseSepTrain[d,l] = mean((y - onePredict[,l])^2)
  }
}

# Calculate the inverse hyperbolic tangent of r
zScores = atanh(sqrt(rsqSepTrain))

# Calculate averaged z value and convert back to r-squared
# Calculate averaged mse
for(r in 1:5){
  Fit[1,r+1] = tanh(mean(zScores[,r]))^2
  Fit[2,r+1] = mean(mseSepTrain[,r])
}

##############################################
# Test Data
##############################################

####### Calculate Fit Statistics for 5 LASSOs ########

# Calculate predicted scores for each participant in training set based on the averaged coefficients
x = data.matrix(imputedTest[,2:50]) # select all columns of predictors
y = as.numeric(imputedTest$depression_raw) # select outcome
predTestSep = matrix(NA,nrow(imputedTest),6) # create empty matrix to store predicted values
for(j in 2:6){
  b0 = Coefficients[1,j]
  b = Coefficients[2:50,j]
  predTestSep[1:nrow(imputedTest),j] = x%*%b+b0 # calculate predicted y scores based on model
}

predTestSep[,1] <- imputedTest$.imp

# Calculate fit statistics within each imputed data set using the final model
rsqSepTest = matrix(NA,m,5)
mseSepTest = matrix(NA,m,5)
for(d in 1:m){
  # Create observed variable for one imputed data set
  oneImputed <- imputedTest[which(imputedTest$.imp==d),-1]
  y <- as.numeric(oneImputed$depression_raw)
  
  # Create predicted scores for one imputed data set
  onePredict <- predTestSep[which(predTestSep[,1]==d),-1]
  
  # Calculate R-squared in training set for all 5 lambda values
  # Calculate MSE, mean of Equation 1 in paper
  for(l in 1:5){
    rsqSepTest[d,l] = 1 - sum((y - onePredict[,l])^2)/sum((y - mean(y))^2)
    mseSepTest[d,l] = mean((y - onePredict[,l])^2)
  }
}

# Calculate the inverse hyperbolic tangent of r
zScoresTest = atanh(sqrt(rsqSepTest))

# Calculate averaged z value and convert back to r-squared
# Calculate averaged mse
for(r in 1:5){
  Fit[3,r+1] = tanh(mean(zScoresTest[,r]))^2
  Fit[4,r+1] = mean(mseSepTest[,r])
}


############################################################################
############################################################################
# Stacked
############################################################################
############################################################################

##############################################
# Prepare data
##############################################

# Remove original data set and only keep imputed data sets, remove ".imp" variable
stackedTrain <- TrainData[TrainData$.imp!=0,-1]
stackedTest <- TestData[TestData$.imp!=0,-1]

# Select original data set (contains missing values) and remove "imp" column
originalTrain <- TrainData[TrainData$.imp==0,-1]
originalTest <- TestData[TestData$.imp==0,-1]

# Create empty weight matrices for one imputed data set
Weights_one_Train = matrix(NA,Ntrain)
Weights_one_Train2 = matrix(NA,Ntrain)
Weights_one_Test = matrix(NA,Ntest)
Weights_one_Test2 = matrix(NA,Ntest)

# Create 2 sets of weights; for each individual, count predictors that are complete
for(i in 1:Ntrain){
  Weights_one_Train[i] = sum(!is.na(originalTrain[i,1:49]))/(P*m)
  Weights_one_Train2[i] = 1/m
}

for(i in 1:nrow(originalTest)){
  Weights_one_Test[i] = sum(!is.na(originalTest[i,1:49]))/(P*m)
  Weights_one_Test2[i] = 1/m
}

# Apply weights to all imputed data sets (participants are in same order for each imputed data set)
WeightsTrain = rep(Weights_one_Train,times=m)
WeightsTrain2 = rep(Weights_one_Train2,times=m)
WeightsTest = rep(Weights_one_Test,times=m)
WeightsTest2 = rep(Weights_one_Test2,times=m)

##############################################
# Cross-validation
##############################################

# 10-fold cross-validation to determine optimal lambda via MSE
set.seed(1989)
cv.stackedW <- cv.glmnet(x=data.matrix(stackedTrain[,1:P]),y=stackedTrain$depression_raw,
                        alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k,weights=WeightsTrain)

# Save value of lambda that gives minimum cvm (mean cross-validated error)
lambdastackedW <- cv.stackedW$lambda.min
Fit[5,7] <- lambdastackedW

# 10-fold cross-validation to determine optimal lambda via MSE using consistent weights
set.seed(1989)
cv.stackedW2 <- cv.glmnet(x=data.matrix(stackedTrain[,1:P]),y=stackedTrain$depression_raw,
                          alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k,weights=WeightsTrain2)

# Save value of lambda that gives minimum cvm (mean cross-validated error)
lambdastackedW2 <- cv.stackedW2$lambda.min
Fit[5,8] <- lambdastackedW2
  
# 10-fold cross-validation to determine optimal lambda via MSE not using weights
set.seed(1989)
cv.stackedNW <- cv.glmnet(x=data.matrix(stackedTrain[,1:P]),y=stackedTrain$depression_raw,
                          alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)

# Save value of lambda that gives minimum cvm (mean cross-validated error)
lambdastackedNW <- cv.stackedNW$lambda.min
Fit[5,9] <- lambdastackedNW

##############################################
# Training Model - Weights
##############################################

# LASSO on full training data using cross-validated lambda
train.stacked = glmnet(x=data.matrix(stackedTrain[,1:P]),y=stackedTrain$depression_raw,
                       alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdastackedW,weights=WeightsTrain)

# Save regression coefficients including intercept
Coefficients[2:nrow(Coefficients),7] <- as.array(train.stacked$beta)
Coefficients[1,7] <- unname(train.stacked$a0)

# Calculate predicted scores for each participant based on the model
predict_stacked_train <- predict(train.stacked,newx=data.matrix(stackedTrain[,1:P]),s=lambdastackedW,type="link")

# Calculate R-squared in training set (incorporates weights)
Fit[1,7] <- train.stacked$dev.ratio

# Calculate MSE, mean of Equation 1 in paper
Fit[2,7] <- mean((stackedTrain$depression_raw - predict_stacked_train)^2)

##############################################
# Training Model - Weights (1/m)
##############################################

# LASSO on full training data using cross-validated lambda
train.stackedW2 = glmnet(x=data.matrix(stackedTrain[,1:P]),y=stackedTrain$depression_raw,
                       alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdastackedW2,weights=WeightsTrain2)

# Save regression coefficients including intercept
Coefficients[2:nrow(Coefficients),8] <- as.array(train.stackedW2$beta)
Coefficients[1,8] <- unname(train.stackedW2$a0)

# Calculate predicted scores for each participant based on the model
predict_stacked_train2 <- predict(train.stackedW2,newx=data.matrix(stackedTrain[,1:P]),s=lambdastackedW2,type="link")

# Calculate R-squared in training set (incorporates weights)
Fit[1,8] <- train.stackedW2$dev.ratio

# Calculate MSE, mean of Equation 1 in paper
Fit[2,8] <- mean((stackedTrain$depression_raw - predict_stacked_train2)^2)

##############################################
# Training Model - No Weights
##############################################

# LASSO on full training data using cross-validated lambda
train.stackedNW = glmnet(x=data.matrix(stackedTrain[,1:P]),y=stackedTrain$depression_raw,
                       alpha=1,standardize=TRUE,intercept=TRUE,lambda=lambdastackedNW)

# Save regression coefficients including intercept
Coefficients[2:nrow(Coefficients),9] <- as.array(train.stackedNW$beta)
Coefficients[1,9] <- unname(train.stackedNW$a0)

# Calculate predicted scores for each participant based on the model
predict_stackedNW_train <- predict(train.stackedNW,newx=data.matrix(stackedTrain[,1:P]),s=lambdastackedNW,type="link")

# Calculate R-squared in training set (incorporates weights)
Fit[1,9] <- train.stackedNW$dev.ratio

# Calculate MSE, mean of Equation 1 in paper
Fit[2,9] <- mean((stackedTrain$depression_raw - predict_stackedNW_train)^2)

##############################################
# Test Model - Weights
##############################################

# Calculate predicted scores for each participant in test set based on training model (weights not needed)
predict_stacked_test <- predict(train.stacked,newx=data.matrix(stackedTest[,1:P]),s=lambdastackedW,type="link")

# Calculate R-squared in test set
rss_stacked_test <- sum((stackedTest$depression_raw - predict_stacked_test)^2)
tss_stacked_test <- sum((stackedTest$depression_raw - mean(stackedTest$depression_raw))^2)
Fit[3,7] <- 1 - rss_stacked_test/tss_stacked_test

# Calculate MSE, mean of Equation 1 in paper
Fit[4,7] <- mean((stackedTest$depression_raw - predict_stacked_test)^2)

##############################################
# Test Model - Weights (1/m)
##############################################

# Calculate predicted scores for each participant in test set based on training model (weights not needed)
predict_stacked_testW2 <- predict(train.stackedW2,newx=data.matrix(stackedTest[,1:P]),s=lambdastackedW2,type="link")

# Calculate R-squared in test set
rss_stacked_testW2 <- sum((stackedTest$depression_raw - predict_stacked_testW2)^2)
tss_stacked_testW2 <- sum((stackedTest$depression_raw - mean(stackedTest$depression_raw))^2)
Fit[3,8] <- 1 - rss_stacked_testW2/tss_stacked_testW2

# Calculate MSE, mean of Equation 1 in paper
Fit[4,8] <- mean((stackedTest$depression_raw - predict_stacked_testW2)^2)

##############################################
# Test Model - No Weights
##############################################

# Calculate predicted scores for each participant in test set based on training model (weights not needed)
predict_stackedNW_test <- predict(train.stackedNW,newx=data.matrix(stackedTest[,1:P]),s=lambdastackedNW,type="link")

# Calculate R-squared in test set
rss_stackedNW_test <- sum((stackedTest$depression_raw - predict_stackedNW_test)^2)
tss_stackedNW_test <- sum((stackedTest$depression_raw - mean(stackedTest$depression_raw))^2)
Fit[3,9] <- 1 - rss_stackedNW_test/tss_stackedNW_test

# Calculate MSE, mean of Equation 1 in paper
Fit[4,9] <- mean((stackedTest$depression_raw - predict_stackedNW_test)^2)


############################################################################
############################################################################
# MI-LASSO
############################################################################
############################################################################

############################################################################
# modified MI.LASSO function
# see Chen and Wang (2013) for original function: http://www.columbia.edu/~qc2138/
############################################################################

MI.LASSO.mod = function(mydata, D, lambda, maxiter=200, eps=1e-6) {
  ## D is the number of imputations
  ## mydata is in the array format: mydata[[1]] is the first imputed dataset...
  ## for each mydata[[d]], the first p columns are covariates X, and the last one is the outcome Y
  
  ## number of observations
  n = dim(mydata[[1]])[1]
  
  ## number of covariates
  p = dim(mydata[[1]])[2]-1
  
  ## Standardize covariates X and center outcome Y 
  x = NULL
  y = NULL
  meanx = NULL
  normx = NULL
  meany = 0
  xOrig = NULL
  yOrig = NULL
  for (d in 1:D) {
    x[[d]] = mydata[[d]][,1:p]
    y[[d]] = mydata[[d]]$depression_raw # specified outcome variable
    meanx[[d]] = apply(x[[d]], 2, mean)
    x[[d]] = scale(x[[d]], meanx[[d]], FALSE)
    normx[[d]] = sqrt(apply(x[[d]]^2, 2, sum))
    x[[d]] = scale(x[[d]], FALSE, normx[[d]])
    meany[d] = mean(y[[d]])
    y[[d]] = y[[d]] - meany[d]
    xOrig[[d]] = mydata[[d]][,1:p]
    xOrig[[d]] = scale(xOrig[[d]],FALSE,FALSE)
    yOrig[[d]] = mydata[[d]]$depression_raw
  }
  
  ## Estimate beta_dj in Equation (4) 
  iter = 0
  dif = 1
  c = rep(1,p)
  b = matrix(0,p,D) 						
  while (iter<=maxiter & dif>=eps) {
    iter = iter + 1
    b.old = b
    # update beta_d by ridge regression
    for (d in 1:D) {
      xtx = t(x[[d]])%*%x[[d]]
      diag(xtx) = diag(xtx) + lambda/c
      xty = t(x[[d]])%*%y[[d]]
      b[,d] = qr.solve(xtx)%*%xty
    } 		
    # update c in Equation (4)
    c = sqrt(apply(b^2,1,sum)) 			
    c[c<sqrt(D)*1e-10] = sqrt(D)*1e-10
    dif = max(abs(b-b.old))
  }
  b[apply((b^2),1,sum)<=5*1e-8,] = 0
  
  ## transform beta and intercept back to original scale
  b0 = 0
  coefficients = matrix(0,p+1,D)
  for (d in 1:D) {
    b[,d] = b[,d]/normx[[d]]
    b0[d] = meany[d] - b[,d]%*%meanx[[d]]
    coefficients[,d] = c(b0[d], b[,d])
  }
  
  ## output the selected variables, TRUE means "selected", FALSE means "NOT selected"
  varsel = abs(b[,1])>0
  
  # average coefficients across imputations to obtain final model
  q = p+1
  bavg = matrix(NA,q)
  for (i in 1:q) {
    bavg[i] = mean(coefficients[i,1:D])
  }
  
  b0_avg = bavg[1]
  b_avg = matrix(NA,p,1)
  b_avg[1:p,1] = bavg[2:q]
  
  # Calculate fit statistics within each imputed data set using the final model
  rsqMIL = matrix(NA,D)
  mseMIL = matrix(NA,D)
  x = NULL
  y = NULL
  yhat = NULL
  for(d in 1:D){
    # Create observed variable for one imputed data set
    y[[d]] = mydata[[d]]$depression_raw
    
    # Create independent variable scores for one imputed data set
    x[[d]] = as.matrix(mydata[[d]][,1:p])
    
    # Create predicted scores for one imputed data set
    yhat[[d]] = x[[d]]%*%b_avg[,1]+b0_avg
    #yhat[[d]] = x[[d]]%*%coefficients[2:(p+1),d]+coefficients[1,d]
    # Use final model, not imputation-specific models
    
    # Calculate R-squared
    rsqMIL[d] = 1 - sum((y[[d]] - yhat[[d]])^2)/sum((y[[d]] - mean(y[[d]]))^2)
    
    # Calculate MSE, mean of Equation 1 in paper
    mseMIL[d] = mean((y[[d]] - yhat[[d]])^2)
  }
  
  # Calculate the inverse hyperbolic tangent of r
  zScoresMIL = atanh(sqrt(rsqMIL))
  
  # Calculate averaged z value and convert back to r-squared
  r2 = tanh(mean(zScoresMIL))^2
  
  # Calculate averaged mse
  mse = mean(mseMIL)

  return(list(coefficients=coefficients, b0 = b0_avg, b = b_avg, mse=mse, r2=r2, varsel=varsel))
}

##############################################
# Prepare data
##############################################

## mydata is in the array format: mydata[[1]] is the first imputed data set...
## for each mydata[[d]], the first p columns are covariates X, and the last one is the outcome Y

# Remove original data set
miLASSOTrain <- TrainData[TrainData$.imp!=0,]
miLASSOTest <- TestData[TestData$.imp!=0,]

##############################################
# Cross-Validation
##############################################

# Create k equally sized folds
fold <- cut(seq(1,Ntrain),breaks=k,labels=FALSE)
folds = rep(fold,times=m)

# Create vector of lambdas to test
# ran the following sequences to narrow down values (save time)
# grid = 10^seq(-3,3,by=1)
# grid = 10^seq(0,2,by=.25)
# grid = 10^seq(1.05,1.5,by=.075)
grid = 10^seq(1.35,1.425,by=.075)
err = matrix(NA,k,length(grid)) # k by lambda

# calculate start time to determine how long analyses take to run, takes a little over 3 hours
start.time.mil <- Sys.time()

# Perform 10-fold cross validation
for(f in 1:k){
  testIndexes <- which(folds==f,arr.ind=TRUE) # Select row indicators for fold f
  trainData <- miLASSOTrain[-testIndexes,] # Create data set for the other 9 folds
  # Create data set that MI-LASSO can read
  # To use MI-LASSO, data is in the array format: data[[1]] is the first imputed data set, data[[2]] is the second, etc.
  NineFoldData <- NULL
  for(d in 1:m){
    NineFoldData[[d]] <- trainData[which(trainData$.imp==d),-1]
  }
  validData <- miLASSOTrain[testIndexes,] # Create data set for f fold (validation set)
  # For 9 of the folds, train the model for a particular lambda value
  for (l in 1:length(grid)){
    fit.train = MI.LASSO.mod(NineFoldData, D=m, lambda=grid[l])
    
    # average the coefficients
    AvgCoef = matrix(NA,P+1)
    for(i in 1:nrow(AvgCoef)){
      AvgCoef[i] <- mean(fit.train$coefficients[i,1:m])
    }
    
    # create empty matrix for MSE values for this fold and lambda value
    mseMIL = matrix(NA,m)
    
    # Calculate predicted scores based on model in validation fold, for each imputed data set
    for (d in 1:m){
      x = as.matrix(validData[which(validData$.imp==d),2:50]) # select all p predictor columns
      y = as.matrix(validData[which(validData$.imp==d),51])
      b0 = AvgCoef[1]
      b = AvgCoef[2:50]
      predictVal = x%*%b+b0
      # Calculate mse in 10th fold
      mseMIL[d] = mean((y - predictVal)^2)
    }
    
    # Take the average mse value across imputed data sets
    # Save averaged mse value for that fold and for that lambda value
    err[f,l] = mean(mseMIL)
  }
}

# calculate end time to determine how long cross-validation takes to run
end.time.mil <- Sys.time()
time.taken.mil <- end.time.mil - start.time.mil

# Calculate mean of the mse values across the k folds (only differ due to sampling error)
mean_err = colMeans(err, na.rm=T)

# Determine the value of lambda that minimized the mse
lambdaMiLASSO <- grid[which(colMeans(err, na.rm=T) == min(colMeans(err, na.rm=T)), arr.ind = TRUE)]

# Save optimal lambda value
Fit[5,10] <- lambdaMiLASSO

##############################################
# Training Model
##############################################

# Prepare data
miLASSOTrainList <- NULL
for(d in 1:m){
  miLASSOTrainList[[d]] <- miLASSOTrain[which(miLASSOTrain$.imp==d),-1]
}

# LASSO on full training data using cross-validated lambda
train.miLASSO = MI.LASSO.mod(miLASSOTrainList, D=m, lambda=lambdaMiLASSO)

# Save averaged regression coefficients including intercept
Coefficients[1,10] <- train.miLASSO$b0
Coefficients[2:nrow(Coefficients),10] <- train.miLASSO$b

# Save MSE and R-squared for training model
Fit[1,10] <- train.miLASSO$r2
Fit[2,10] <- train.miLASSO$mse

##############################################
# Test Model
##############################################

rsqMILTest = matrix(NA,m)
mseMILtest = matrix(NA,m)

# Calculate predicted scores and fit stats in test data using training model for each imputed data set
for (d in 1:m){
  x = as.matrix(miLASSOTest[which(miLASSOTest$.imp==d),2:50]) # select all p predictor columns
  y = as.matrix(miLASSOTest[which(miLASSOTest$.imp==d),51])
  b0 = train.miLASSO$b0
  b = train.miLASSO$b
  predictTest = x%*%b+b0
  # Calculate r-squared
  rsqMILTest[d] = 1 - sum((y - predictTest)^2)/sum((y - mean(y))^2)
  # Calculate mse in test set
  mseMILtest[d] = mean((y - predictTest)^2)
}

# Calculate the inverse hyperbolic tangent of r
zScoresMILTest = atanh(sqrt(rsqMILTest))

# Calculate averaged z value and convert back to r-squared
# Calculate averaged mse
Fit[3,10] = tanh(mean(zScoresMILTest))^2
Fit[4,10] = mean(mseMILtest)



##########################################################################################################
##########################################################################################################
# Summarizing Results from Three Approaches
##########################################################################################################
##########################################################################################################

# Apply variable names to rows of Coefficients matrix
rownames(Coefficients) <- VarNames

# Number of variables chosen (not including intercept)
VarChosen = matrix(NA,10)
for(c in 1:10){
  VarChosen[c] = sum(abs(Coefficients[2:nrow(Coefficients),c])>0)
}
# Select only columns of methods presented in paper
VarChosen[c(1,2,9,10)]

Fit[,c(1,2,9,10)]

# How many methods chose this variable?
Coefficients.mod <- Coefficients[,c(2,9,10)]
MethChosen = matrix(NA,P+1)
for(p in 1:(P+1)){
  MethChosen[p] = sum(abs(Coefficients.mod[p,1:ncol(Coefficients.mod)])>0)
}
rownames(MethChosen) <- VarNames

# Number of variables that all 3 approaches chose 
sum(MethChosen[,1] == 3) - 1 # subtract 1 for intercept

# Reorder variables
row_order <- c("Intercept","age","audit_c","count_lf","ptsd","anxietyr","sadr","intfsocr","energr",
               "anxiety_raw","srMentH","srPhysH","srLivSit","srDrugFree","srSocNet","srSexRx","ssHelp",
               "ssEmoHelp","ssNewFri","smAppFreq","datingAppFreq","losAng","Black","Latinx","White","Other","female","cis",
               "heterosexual","employed","poverty","insured","healthProvider","medicalUtil_re","erCare_re",
               "sapro_lf","hivpp_lf","homeless_lf","incarceration_lf","ipv_lf","sexex_lf","suicide_lf",
               "hospitalmh_lf","sexab_lf","tmx5old_lf","tmrob_lf","tmotinj_lf","tmotmdr_lf","xdrug_r","smoke_lf")

# Create table of coefficients and fit statistics, Round decimal places
Table2a <- round(Coefficients[,c(1:10)],2)
Table2a <- Table2a[row_order,]
colnames(Table2a)<-c("Listwise","SepAvg","SepMed","SepMin","SepMax","SepInd",
                     "StackedW","StackedW2","StackedNW","MI-LASSO")
Table2b <- round(Coefficients[,c(1,2,9,10)],2)
Table2b <- cbind(Table2b[,1:2],InclFreq[,1],Table2b[,3:4])
Table2b <- Table2b[row_order,]
colnames(Table2b)<-c("Listwise","SepMax","IF","StackedNW","MI-LASSO")

# Round decimal places
Fit_round <- round(Fit,3)
Fit_roundb <- Fit_round[,c(1,2,9,10)]

# Output coefficients and fit statistics as csv files
#write.csv(Table2b, paste0(WRKDIR,"Table2_Coefficients.csv"))
#write.csv(Fit_roundb, paste0(WRKDIR,"Table2_Fit.csv"))

################################################################
# FMI
################################################################

# Convert imputed data set to mids type
TrainData_mids<-as.mids(TrainData)

# Analyze an OLS Regression within each imputed data set (remove Latinx dummy code to match stacked approach)
OLSestimates <- with(TrainData_mids,
                     lm(depression_raw ~ age+audit_c+count_lf+ptsd+anxietyr+sadr+intfsocr+energr+
                          anxiety_raw+srMentH+srPhysH+srLivSit+srDrugFree+srSocNet+srSexRx+ssHelp+
                          ssEmoHelp+ssNewFri+smAppFreq+datingAppFreq+losAng+female+cis+heterosexual+
                          employed+poverty+insured+healthProvider+medicalUtil_re+erCare_re+sapro_lf+
                          hivpp_lf+homeless_lf+incarceration_lf+ipv_lf+sexex_lf+suicide_lf+hospitalmh_lf+
                          sexab_lf+tmx5old_lf+tmrob_lf+tmotinj_lf+tmotmdr_lf+xdrug_r+smoke_lf+Black+
                          White+Other))

# Pool the estimates and standard errors via Rubin's Rules
OLSresults <- summary(pool(OLSestimates))
olsPooled <- pool(OLSestimates)
fmiOLS <- olsPooled$pooled$fmi

# Select original data set (contains missing values) and remove ".imp" and ".id" columns
original <- fullDataID[fullDataID$.imp==0,-c(1,52)]

# Calculate number of missing values per variable as well as percentage, format n (%)
datmiss <- sapply(original,function(x) sum(is.na(x)))
percmiss <- round(datmiss/nrow(original)*100,2)
Table <- cbind(names(original),percmiss)

# Reorder to match Inclusion Frequency
Table <- Table[c(50,1:49),]

# Reorder FMI variable from OLS to match Inclusion Frequency (Latinx was reference variable)
fmiOLS_r <- fmiOLS[c(1:47,NA,48,49)]

# Merge with Inclusion Frequency data and FMI variable from OLS
IFfmi <- cbind(InclFreq,Table[,2],fmiOLS_r)
colnames(IFfmi)[6] <- "PercentMissing"
colnames(IFfmi)[7] <- "FMI"
IFfmi$PercentMissing <- as.numeric(IFfmi$PercentMissing)

# Remove intercept and latinx rows
IFfmi <- IFfmi[c(2:47,49,50),]

# Create variable that measures consistent selection
IFfmi$consistent <- ifelse(IFfmi$Max==100|IFfmi$Max==0,1,0)
table(IFfmi$consistent)

# Scatterplot
plot(IFfmi$FMI,IFfmi$Max,xlab="FMI",ylab="Inclusion Frequency - Max Lambda")
plot(IFfmi$PercentMissing,IFfmi$Max)

# Look at means of FMI by consistency of selection
aggregate(IFfmi$FMI, list(IFfmi$consistent), mean)
aggregate(IFfmi$PercentMissing, list(IFfmi$consistent), mean)

# Calculating means without 1 outlier
Outlier <- IFfmi[IFfmi$PercentMissing <= 10,]
aggregate(Outlier$FMI, list(Outlier$consistent), mean)
aggregate(Outlier$PercentMissing, list(Outlier$consistent), mean)

################################################################
# PSR
################################################################

# Remove duplicated column and only keep Y imputation model
# reorder rows to match lasso results (residual variance instead of Black dummy variable)
psr_r <- psr[psr$parm <= 50,c(1,3:5)]
order <- c(1:22,48:50,23:47)
psr_r <- cbind(psr_r, order)

psr_r <- psr_r[order(psr_r$i,psr_r$order),]

# Merge psr values with inclusion frequencies
IFpsr <- cbind(InclFreq,psr_r)

# Group variables into those selected and those not based on threshold
IFpsr$selected <- ifelse(IFpsr$Mean >= 50,1,0)

# Group variables based on consistency of selection
IFpsr$consistent <- ifelse(IFpsr$Mean==100|IFpsr$Mean==0,1,0)

# Look at only first few imputations and remove intercept
IFpsr2 <- IFpsr[2:50,]
table(IFpsr2$consistent)

aggregate(IFpsr2[,7], list(IFpsr2$consistent), mean)

# Scatterplots
plot(IFpsr2[,7],IFpsr2[,12], xlab="PSRF", ylab="Consistent")
plot(IFpsr2[,7],IFpsr2[,4], xlab="PSRF", ylab="Inclusion Frequency")

#Regression of consistent selection on PSRF (as PSRF increases, IF decreases, not significant)
Reg1 <- glm(consistent ~ PSR, family=binomial(link='logit'), data=IFpsr2)
summary(Reg1)

##############################################
# Relative magnitudes of the slopes when they get selected
##############################################

## transform betas from m imputed data sets to standardized scale (mean lambda)
bSTD = matrix(0,P,m)
stdX = matrix(0,P,1)
for (d in 1:m) {
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  x <- data.matrix(oneImputed[,1:49])
  y <- as.numeric(oneImputed$depression_raw)
  b <- as.numeric(coefSeparMax[2:50,d])
  for (p in 1:P){
    xVar = x[,p] # Selecting just one x variable
    stdX[p] = sd(xVar)
    bSTD[p,d] = b[p]*stdX[p]/sd(y)
  }
}

# Select inclusion frequency of just the mean lambda and merge with standardized coefficients
IncFreq <- InclFreq[2:nrow(InclFreq),1]
CoefIF <- cbind(bSTD,IncFreq)

sd = NA
min = NA
max = NA
range = NA
minabs = NA
maxabs = NA
for(j in 1:(P)){
  row = CoefIF[j,1:50]
  nonzero = row[row!=0]
  if(length(nonzero)==0){
    sd[j] = NA
    min[j] = NA
    max[j] = NA
    range[j] = NA
    minabs[j] = NA
    maxabs[j] = NA
  } else {
    sd[j] = sd(nonzero)
    min[j] = min(nonzero)
    max[j] = max(nonzero)
    range[j] = abs(max[j] - min[j])
    minabs[j] = min(abs(nonzero))
    maxabs[j] = max(abs(nonzero))
  }
}

StdSlopes <- cbind(CoefIF,sd,min,max,range,minabs,maxabs)

# Apply variable names to rows of Coefficients matrix
rownames(StdSlopes) <- VarNames[2:50]
View(StdSlopes)

# Examples
StdSlopes[9,c(51,53:54)] # anxiety_raw
StdSlopes[45,c(51,53:54)] # smoke_lf

#for(j in 1:P){
#  hist(StdSlopes[j,1:50], xlab="Coefficients")
#}

##############################################
# Plots of lambda values
##############################################

layout(matrix(c(1,2,3,4,5,5),ncol=2,byrow=TRUE),heights=c(3,3,1))

par(mar=c(4,4,3,2))

# Save all m optimal lambda values
lambdaSeparateCheck = matrix(NA,m)

# Run k-fold cross validation on each imputed data set independently and save optimal lambda value
for(d in 1:m){
  # Create one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  
  # 10-fold Cross-validation to find lambda (for that imputed data set)
  set.seed(17463) # semi-normal
  cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                           alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
  
  # save value of lambda that gives minimum cvm (mean cross-validated error) for that imputed data set
  lambdaSeparateCheck[d] <- cv.separate$lambda.min
}

# Distribution of lambda values
hist(lambdaSeparateCheck, xlab="Lambda",main="",ylim = c(0,25),xlim = c(.06,.17))

# Save all m optimal lambda values
lambdaSeparateCheck = matrix(NA,m)

# Run k-fold cross validation on each imputed data set independently and save optimal lambda value
for(d in 1:m){
  # Create one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  
  # 10-fold Cross-validation to find lambda (for that imputed data set)
  set.seed(48766)
  cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                           alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
  
  # save value of lambda that gives minimum cvm (mean cross-validated error) for that imputed data set
  lambdaSeparateCheck[d] <- cv.separate$lambda.min
}

# Distribution of lambda values
hist(lambdaSeparateCheck, xlab="Lambda",main="",ylim = c(0,25),xlim = c(.06,.17))

# Save all m optimal lambda values
lambdaSeparateCheck = matrix(NA,m)

# Run k-fold cross validation on each imputed data set independently and save optimal lambda value
for(d in 1:m){
  # Create one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  
  # 10-fold Cross-validation to find lambda (for that imputed data set)
  set.seed(879327) # positively skewed
  cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                           alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
  
  # save value of lambda that gives minimum cvm (mean cross-validated error) for that imputed data set
  lambdaSeparateCheck[d] <- cv.separate$lambda.min
}

# Distribution of lambda values
hist(lambdaSeparateCheck, xlab="Lambda",main="",ylim = c(0,25),xlim = c(.06,.17))

# Save all m optimal lambda values
lambdaSeparateCheck = matrix(NA,m)

# Run k-fold cross validation on each imputed data set independently and save optimal lambda value
for(d in 1:m){
  # Create one imputed data set
  oneImputed <- imputedTrain[which(imputedTrain$.imp==d),-1]
  
  # 10-fold Cross-validation to find lambda (for that imputed data set)
  set.seed(97781)
  cv.separate <- cv.glmnet(x=data.matrix(oneImputed[,1:P]), y=oneImputed$depression_raw,
                           alpha=1,standardize=TRUE,intercept=TRUE,type.measure="mse",nfolds=k)
  
  # save value of lambda that gives minimum cvm (mean cross-validated error) for that imputed data set
  lambdaSeparateCheck[d] <- cv.separate$lambda.min
}

# Distribution of lambda values
hist(lambdaSeparateCheck, xlab="Lambda",main="",ylim = c(0,25),xlim = c(.06,.17))


# Save all m optimal lambda values
lambdaSeparateCheck = matrix(NA,m)

##############################################
# Examples of pooled estimates in paper
##############################################

InclFreq["anxiety_raw",1]
InclFreq["ssNewFri",1]
InclFreq["smoke_lf",1]
InclFreq["age",1]

Table3 <- t(coefSeparAvg[c(10,19,46,2),])
colnames(Table3)<-c("anxiety_raw","ssNewFri","smoke_lf","age")
# rows are imputed data sets

# Calculate mean of slopes regardless of threshold
meanslopes = matrix(NA,1,4)
for (a in 1:4){
  meanslopes[1,a] <- mean(Table3[1:50,a])
}
Table3 <- rbind(Table3,meanslopes)

# Round decimal places
Table3 <- round(Table3,3)

View(Table3)

# Output coefficients as csv file
#write.csv(Table3, paste0(WRKDIR,"Table3_Coefficients.csv"))

# Add twos row for inclusion frequency and estimate used in final model, which matches Table 2

##############################################
# Calculate end time to determine how long analyses take to run
##############################################

end.time.total <- Sys.time()
time.taken.total <- end.time.total - start.time.total
time.taken.total
time.taken.mil
length(grid)
time.taken.sep