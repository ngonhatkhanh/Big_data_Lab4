from typing import List
import numpy as np
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when

class SVM:
    def __init__(self, max_iter=100, reg_param=0.1):
        self.model = LinearSVC(
            maxIter=max_iter,
            regParam=reg_param,
            labelCol="label",
            featuresCol="features",
            weightCol="weight"
        )
        self.fitted_model = None

    def train(self, df: DataFrame) -> List:
        if df.count() == 0:
            return [], 0.0, 0.0, 0.0, 0.0

        # Kiểm tra số lượng nhãn (phải là 2 cho binary classification)
        num_labels = df.select("label").distinct().count()
        if num_labels != 2:
            raise ValueError(f"Expected 2 labels for binary classification, but found {num_labels}")

        # Tính trọng số lớp để xử lý imbalance
        label_counts = df.groupBy("label").count().collect()
        total = df.count()
        weights = {row['label']: total / (2 * row['count']) for row in label_counts}

        # Ánh xạ trọng số vào cột 'weight' sử dụng when
        df = df.withColumn(
            "weight",
            when(col("label") == list(weights.keys())[0], list(weights.values())[0])
            .otherwise(list(weights.values())[1])
        )

        # Huấn luyện mô hình
        self.fitted_model = self.model.fit(df)
        self.fitted_model.write().overwrite().save("svm_model")

        # Dự đoán
        predictions = self.fitted_model.transform(df)

        # Tính số liệu đánh giá
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        )
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        )
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        accuracy = accuracy_evaluator.evaluate(predictions)
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)

        prediction_list = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
        return prediction_list, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        if df.count() == 0:
            return [], 0.0, 0.0, 0.0, 0.0, np.zeros((2, 2))

        # Kiểm tra số lượng nhãn (phải là 2 cho binary classification)
        num_labels = df.select("label").distinct().count()
        if num_labels != 2:
            raise ValueError(f"Expected 2 labels for binary classification, but found {num_labels}")

        # Dự đoán
        predictions = self.fitted_model.transform(df)

        # Tính số liệu đánh giá
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        )
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        )
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        accuracy = accuracy_evaluator.evaluate(predictions)
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)

        prediction_list = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

        # Tính confusion matrix cho binary classification
        cm = predictions.groupBy("label", "prediction").count().groupBy("label").pivot("prediction").sum("count").fillna(0).collect()
        cm_matrix = np.zeros((2, 2))
        label_map = sorted([row['label'] for row in df.select("label").distinct().collect()])  # [0, 1] hoặc [1, 0]

        for row in cm:
            label_idx = label_map.index(row['label'])
            for pred in label_map:
                pred_idx = label_map.index(pred)
                cm_matrix[label_idx][pred_idx] = row[str(pred)] if str(pred) in row else 0

        return prediction_list, accuracy, precision, recall, f1, cm_matrix