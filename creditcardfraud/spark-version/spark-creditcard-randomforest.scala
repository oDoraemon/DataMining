import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassifier, GBTClassifier}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// 数据预处理
val file_path = "hdfs://localhost:9000/data/creditcard.csv"
val data = spark.read.option("header", true).option("inferSchema", true).csv(file_path)
val colNames = data.columns.filter(_ != "Class")
val assembler = new VectorAssembler().setInputCols(colNames).setOutputCol("featureVector")
val indexedFeatures = new VectorIndexer().setInputCol("featureVector").setOutputCol("indexedFeatures").setMaxCategories(4)
val indexedLabel = new StringIndexer().setInputCol("Class").setOutputCol("indexedLabel")
val pipe1 = new Pipeline().setStages(Array(assembler, indexedFeatures, indexedLabel))
val rdata = pipe1.fit(data).transform(data)
val data = rdata.select("indexedFeatures", "indexedLabel") // 脚本中不允许重新赋值，但在shell下执行OK
val Array(train, test) = data.randomSplit(Array(0.7, 0.3))
train.cache()
test.cache()

// 模型训练
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
val rf_model = rf.fit(train)
val gbt_model = gbt.fit(train)
val rf_prediction = rf_model.transform(test)
val gbt_prediction = gbt_model.transform(test)


// AUC参数评估
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
val rf_auc = evaluator.evaluate(rf_prediction)
val gbt_auc = evaluator.evaluate(gbt_prediction)

// 建模训练过程中对数据进行了类别转换，得到的prediction需要进行逆转换
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictionLabel").setLabels(labelIndexer.labels)
