import time
import json
import socket
import argparse
import pandas as pd
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='Streams a file to a Spark Streaming Context')
parser.add_argument('--folder', '-f', help='Path to tweet_sentiment.csv', required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream', required=False, type=bool, default=False)
parser.add_argument('--split', '-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep', '-t', help="streaming interval", required=False, type=int, default=5)

TCP_IP = "localhost"
TCP_PORT = 6100

class Dataset:
    def __init__(self) -> None:
        self.tweets = []
        self.sentiments = []

    def data_generator(self, data_file: str, batch_size: int, split: str = 'train'):
        df = pd.read_csv(data_file)
        if df.empty or 'tweet' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV file is empty or missing required columns 'tweet' or 'sentiment'")

        if split == 'train':
            df = df.sample(frac=0.8, random_state=42)
        else:
            df = df.drop(df.sample(frac=0.8, random_state=42).index)

        tweets = df['tweet'].tolist()
        sentiments = df['sentiment'].tolist()

        self.tweets.extend(tweets)
        self.sentiments.extend(sentiments)

        batches = []
        for ix in range(0, len(self.tweets), batch_size):
            tweet_batch = self.tweets[ix:ix + batch_size]
            sentiment_batch = self.sentiments[ix:ix + batch_size]
            batches.append([tweet_batch, sentiment_batch])

        self.tweets = []
        self.sentiments = []
        return batches

    def sendTweetBatchToSpark(self, tcp_connection, input_file, batch_size, split="train"):
        df = pd.read_csv(input_file)
        if split == 'train':
            total_samples = int(len(df) * 0.8)
        else:
            total_samples = int(len(df) * 0.2)
        total_batch = max(1, (total_samples + batch_size - 1) // batch_size)

        pbar = tqdm(total=total_batch)
        data_received = 0

        batches = self.data_generator(input_file, batch_size, split)
        for batch in batches:
            tweets, sentiments = batch

            payload = dict()
            for batch_idx in range(len(tweets)):
                payload[batch_idx] = {
                    'tweet': tweets[batch_idx],
                    'sentiment': sentiments[batch_idx]
                }

            payload = (json.dumps(payload) + "\n").encode()
            try:
                tcp_connection.send(payload)
            except BrokenPipeError:
                print("Connection was closed or batch size too large")
                break
            except Exception as error_message:
                print(f"Error sending batch: {error_message}")
                break

            data_received += 1
            pbar.update(n=1)
            pbar.set_description(f"it: {data_received} | sent: {len(tweets)} tweets")
            time.sleep(sleep_time)

    def connectTCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def streamTweetDataset(self, tcp_connection, folder, batch_size, split):
        input_file = os.path.join(folder, 'tweet_sentiment.csv')
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File {input_file} not found")
        self.sendTweetBatchToSpark(tcp_connection, input_file, batch_size, split)

if __name__ == '__main__':
    args = parser.parse_args()
    data_folder = args.folder
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    try:
        if endless:
            while True:
                dataset.streamTweetDataset(tcp_connection, data_folder, batch_size, train_test_split)
        else:
            dataset.streamTweetDataset(tcp_connection, data_folder, batch_size, train_test_split)
    except KeyboardInterrupt:
        print("Streaming stopped by user")
    finally:
        tcp_connection.close()