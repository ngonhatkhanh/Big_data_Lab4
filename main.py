from pyspark.sql import SparkSession
from model.svm import SVM
from trainer import Trainer, SparkConfig

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("TweetSentiment") \
    .master("local[4]") \
    .getOrCreate()

# Tạo SparkConfig và Trainer
spark_config = SparkConfig()
model = SVM(max_iter=100, reg_param=0.1)
trainer = Trainer(model=model, split="train", spark_config=spark_config, spark_session=spark)
trainer.train()

# Dừng SparkSession
spark.stop()