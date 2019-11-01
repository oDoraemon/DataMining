# Spark 版本的实现
由于Spark.mllib没有AdaBoost的实现，使用Gradient-boosted Tree作为对比。  

RandomForest AUC: 99.947%
Gradient-boosted Tree AUC: 99.953%
Spark.ml 版本的表现比sklearn的实现表现上要优秀很多。造成差异的具体原因待分析。