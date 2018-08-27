library(readr)
library(tidyr)
library(dplyr)
library(caret)
library(purrr)
library(xgboost)


setwd("D:/kaggle")


## Import

test <- read_csv("application_test.csv")
apptrain <- read_csv("application_train.csv")
credit_card <- read_csv("credit_card_balance.csv")
inst_payment <- read_csv("installments_payments.csv")
bureu_bal <- read_csv("bureau_balance.csv")
bureu <- read_csv("bureau.csv")
pos_cash_bal <- read_csv("POS_CASH_balance.csv")
preV_app <- read_csv("previous_application.csv")



## PREPROCESS

bbalance <- bureu_bal %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_BUREAU) %>% 
  summarise_all(mean) 
gc()

bureau <- bureu %>% 
  left_join(bbalance, by = "SK_ID_BUREAU") %>% 
  select(-SK_ID_BUREAU) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(mean)
rm(bbalance, bureu_bal); gc()

credit_card <- credit_card %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(mean)
gc()

inst_payment <- inst_payment %>% 
  select(-SK_ID_PREV) %>% 
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         shortfall = AMT_INSTALMENT - AMT_PAYMENT) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(mean) 
gc()

pos_cash_bal <- pos_cash_bal %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(mean)
 gc()

 preVapp <- preV_app %>%
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
         DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
         DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
         DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
         DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
         APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(mean) 
gc()


rm(bbalance, bureu_bal, bureu, preV_app)
## Joining Data Tables
train_test <- apptrain %>% 
  select(-TARGET) %>%
  bind_rows(test) %>%
  left_join(bureau, by = "SK_ID_CURR") %>% 
  left_join(credit_card, by = "SK_ID_CURR") %>% 
  left_join(inst_payment, by = "SK_ID_CURR") %>% 
  left_join(pos_cash_bal, by = "SK_ID_CURR") %>% 
  left_join(preVapp, by = "SK_ID_CURR") %>% 
  select(-SK_ID_CURR) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED)) %>%
  data.matrix()

trind <-  1:nrow(apptrain)
train_target <- apptrain$TARGET

train <- as.data.frame(train_test)
nzv <- nearZeroVar(train, saveMetrics= TRUE)

train <- train_test[,!nzv$nzv]



xgb.test <- xgb.DMatrix(data = train[-trind, ])     #Creates XBG.Dmatrix obj
tr_val <- train[trind, ]
part <- createDataPartition(train_target, p = 0.9, list = F) %>% c()    #createDataPartition from caret package align test/train partitions
xgb.train <- xgb.DMatrix(data = tr_val[part, ], label = train_target[part])
xgb.valid <- xgb.DMatrix(data = tr_val[-part, ], label = train_target[-part])

rm(tr_val,part); gc()

p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 4,            #number of parallel threads
          eta = 0.05,             #learning rate
          max_depth = 6,          #max depth of a tree
          min_child_weight = 30,
          gamma = 0,              #min loss reduction to make a further partition on a leaf note.
          subsample = 0.85,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.632,
          alpha = 0,                 #L1 regularization term
          lambda = 0,                #L2 regularization term
          nrounds = 2000)            #max number of eta iteration

set.seed =10
xgb.model <- xgb.train(p, xgb.train, p$nrounds, list(val = xgb.valid), print_every_n = 50, early_stopping_rounds = 300)


xgb_pred <- predict(xgb.model, xgb.test)
write.csv(xgb_pred, file = "resulttest.csv")