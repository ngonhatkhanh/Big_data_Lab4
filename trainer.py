import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

class SparkConfig:
    appName = "TweetSentiment"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 2

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split: str, 
                 spark_config: SparkConfig, 
                 spark_session: SparkSession,
                 transforms=None) -> None:
        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf, self.transforms)
        
        self.tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
        self.remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        self.hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
        self.idf = IDF(inputCol="raw_features", outputCol="features")
        self.label_indexer = StringIndexer(inputCol="sentiment", outputCol="label", stringOrderType="frequencyDesc")
        
        self.preprocessing_pipeline = Pipeline(stages=[
            self.tokenizer,
            self.remover,
            self.hashingTF,
            self.idf,
            self.label_indexer
        ])

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)
        self.ssc.start()
        try:
            self.ssc.awaitTermination()
        except KeyboardInterrupt:
            print("Training stopped by user")
        finally:
            self.ssc.stop(stopSparkContext=False, stopGraceFully=True)

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            schema = StructType([
                StructField("tweet", StringType(), True),
                StructField("sentiment", StringType(), True)
            ])
            df = self.sqlContext.createDataFrame(rdd, schema)
            
            if not hasattr(self, 'fitted_pipeline'):
                self.fitted_pipeline = self.preprocessing_pipeline.fit(df)
                self.fitted_pipeline.write().overwrite().save("preprocessing_pipeline")
            preprocessed_df = self.fitted_pipeline.transform(df)
            
            if preprocessed_df.count() > 0:
                predictions, accuracy, precision, recall, f1 = self.model.train(preprocessed_df)
                print("="*10)
                print(f"Predictions = {predictions}")
                print(f"Accuracy = {accuracy:.4f}")
                print(f"Precision = {precision:.4f}")
                print(f"Recall = {recall:.4f}")
                print(f"F1 Score = {f1:.4f}")
                print("="*10)
            else:
                print("Preprocessed DataFrame is empty")
        
        print("Total Batch Size of RDD Received :", rdd.count())
        print("+"*20)