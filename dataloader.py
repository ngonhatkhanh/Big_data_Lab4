import json
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from trainer import SparkConfig

class DataLoader:
    def __init__(self, 
                 sparkContext: SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig, 
                 transforms=None) -> None:
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )
        self.transforms = transforms

    def parse_stream(self) -> DStream:
        def parse_json(line):
            try:
                return json.loads(line) if line.strip() else {}
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
                return {}
        
        json_stream = self.stream.map(parse_json)
        json_stream_exploded = json_stream.flatMap(lambda x: x.values() if x else [])
        tweet_sentiment_stream = json_stream_exploded.map(
            lambda x: (x['tweet'], x['sentiment']) if isinstance(x, dict) and 'tweet' in x and 'sentiment' in x else ("", "")
        )
        tweet_sentiment_stream = tweet_sentiment_stream.filter(lambda x: x[0] and x[1])
        return tweet_sentiment_stream

    @staticmethod
    def preprocess(stream: DStream, transforms=None) -> DStream:
        return stream