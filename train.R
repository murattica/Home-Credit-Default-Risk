library(readr)
library(tidyr)
library(dplyr)
library(skimr)
library(caret)
library(purrr)
library(PCAmixdata)
library(neuralnet)


setwd("D:/kaggle")

apptrain <- read_csv("application_train.csv")
#credit_card <- read_csv("credit_card_balance.csv")
#inst_payment <- read_csv("installments_payments.csv")
#bureu_bal <- read_csv("bureau_balance.csv")
#bureu <- read_csv("bureau.csv")
#pos_cash_bal <- read_csv("POS_CASH_balance.csv")
#preV_app <- read_csv("previous_application.csv")
#pos_cash_bal <- read_csv("POS_CASH_balance.csv")


Y <- apptrain[,2]
X <- apptrain[,c(-1,-2)]

#column data types
sapply(X, class) %>% head(10)

#percantage of NA's in each column
missing_tbl <- X %>%
  summarize_all(.funs = ~ sum(is.na(.)) / length(.)) %>%
  gather() %>%
  arrange(desc(value)) %>%
  filter(value > 0)  

#percantage of default 
perc_of_def <- sum(Y)/length(t(Y))

# number of unique elements in numerical columns
unique_numeric_values_tbl <- X %>%
  select_if(is.numeric) %>%
  map_df(~ unique(.) %>% length()) %>%
  gather() %>%
  arrange(value)

#name of the character columns to convert factors
string_2_factor_names <- X %>%
  select_if(is.character) %>%
  names()

#factorizing "NA" values
str_2fac_X <- X[string_2_factor_names] 
str_2fac_X[is.na(str_2fac_X)]  <- c("NA")
X[string_2_factor_names] <- str_2fac_X




#If the number of unique elements in a numerical column is below this limit, column will be converted to a factor.
factor_limit <- 7         

#convert numeric data to  factor
num_2_factor_names <- unique_numeric_values_tbl %>%   
  filter(value < factor_limit) %>%
  arrange(desc(value)) %>%
  pull(key) %>%
  as.character()

X[num_2_factor_names] <- lapply(X[num_2_factor_names], as.factor)



#PCAmix

#columns with near zero varianca contribution is not included.

X.quan <- as.data.frame(X[,map_lgl(X, is.numeric)]) 
nzv <- nearZeroVar(X.quan, saveMetrics= TRUE) 
X.quan <- X.quan[,!nzv$nzv]

X.qual <-  as.data.frame(X[,!map_lgl(X, is.numeric)])

dimn <- 80  # first n principle components to use in analysis

pca<-PCAmix(X.quan,X.qual,ndim = dimn, rename.level = TRUE,graph=TRUE)

pcVars <- as.data.frame(pca$scores[,1:dimn])
colnames(pcVars) <- paste("X", 1:dimn, sep="")

pcVars <- cbind(pcVars, as.data.frame(Y))


######  glm ######

#create train and test set
ind <- sample(1:dim(pcVars)[1] , 0.8*dim(pcVars)[1]) #indices of train set - use 80% to train
traindf <- pcVars[ind,]
testdf <- pcVars[-ind,]

indvars <- paste("X", 1:dimn, sep="")
form <- as.formula(paste("TARGET ~ ", paste(indvars, collapse= "+")))

glm.fit <- glm(form, data = traindf, family = binomial)
fits <- glm.fit$fitted.values






######  #ANN by neuralNet ######

#ANN configuration
# 150 - 50 - 10 - 1   (two hidden layers)

#form <- as.formula(paste("TARGET ~ ", paste(indvars, collapse= "+")))

#nn <- neuralnet(formula = form,  c(50,10), act.fct = "logistic", data = traindf)












