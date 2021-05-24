'
title: "Honda-Random-Forest Code"
author: "Shohei ONIMARU shohei"
date:2020/08/27
'

### ソースコードの場所をワーキングディレクトリに
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


### ライブラリの読み込み
library(randomForest)

### データの確認
### 1-rawdataというフォルダ内にrawdata.csvというファイルを入れてください
df <- read.csv("rawdata.csv") 
head(df)


### 標準化
df_st = scale(df)
head(df_st)
write.csv(df_st, "df_st.csv")

### ランダムフォレストの実行
model_rf = randomForest(target ~ ., data = df, importance=TRUE) 
model_rf
rf_imp = importance(model_rf)
rf_imp
varImpPlot(model_rf)
file.remove("df_st.csv")

### 重要度の出力
write.csv(rf_imp, "df_importance.csv")


