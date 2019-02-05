## Check predictiveness for the better models by major groups
library(dplyr)
library(magrittr)

factor <- c("Survived", "svmpred", "nbpred",  "rfpred",  "xgbpredt", "xgbttpred", "gbmpred", "voted")
transform <- compareSamp.df %>% mutate_at(vars(factor), funs(as.numeric(as.character(.))))

transform %>% 
  group_by(Sex) %>% 
  dplyr::summarize(Mean = mean(Survived, na.rm=TRUE), rfpred = kappaOnline(Survived, rfpred), 
                   xgbttpred = kappaOnline(Survived, xgbttpred), mxnetfpred = kappaOnline(Survived, mxnetfpred), 
                   bayespred = kappaOnline(Survived, bayespred), voted = kappaOnline(Survived, voted))

transform %>% 
  group_by(Embarked) %>% 
  dplyr::summarize(Mean = mean(Survived, na.rm=TRUE), rfpred = kappaOnline(Survived, rfpred), 
                   xgbttpred = kappaOnline(Survived, xgbttpred), mxnetfpred = kappaOnline(Survived, mxnetfpred), 
                   bayespred = kappaOnline(Survived, bayespred), voted = kappaOnline(Survived, voted))

transform %>% 
  group_by(salutation) %>% 
  dplyr::summarize(Mean = mean(Survived, na.rm=TRUE), rfpred = kappaOnline(Survived, rfpred), 
                   xgbttpred = kappaOnline(Survived, xgbttpred), mxnetfpred = kappaOnline(Survived, mxnetfpred), 
                   bayespred = kappaOnline(Survived, bayespred), voted = kappaOnline(Survived, voted))

transform %>% 
  group_by(deck) %>% 
  dplyr::summarize(Mean = mean(Survived, na.rm=TRUE), rfpred = kappaOnline(Survived, rfpred), 
                   xgbttpred = kappaOnline(Survived, xgbttpred), mxnetfpred = kappaOnline(Survived, mxnetfpred), 
                   bayespred = kappaOnline(Survived, bayespred), voted = kappaOnline(Survived, voted))