from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, StringType

from data_utils import preprocess_data
from model_utils import fit_model, evaluate_model
from pyspark.sql import DataFrame

import argparse
import time
from functools import partial
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_optims', action='store_true')
    parser.add_argument('--data_nodes', type=int, choices=(1, 3), default=1)
    return parser.parse_args()


def draw_stats(times, RAMs, n_nodes, is_optimized):
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))

    optim_str = 'с оптимизацями' if is_optimized else 'без оптимизаций'
    axes[0].hist(times, bins=15)
    axes[0].set_xlabel('Время')
    axes[0].set_ylabel('Частота')
    axes[0].set_title(f'Время: {optim_str}, datanodes={n_nodes}')
    axes[0].grid(True)

    axes[1].hist(RAMs, bins=15)
    axes[1].set_xlabel('RAM(MB)')
    axes[1].set_ylabel('Частота')
    axes[1].set_title(f'RAM: {optim_str}, datanodes={n_nodes}')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots_{n_nodes}_optim_{is_optimized}.png')


def get_executor_memory(sc):
    executor_memory_status = sc._jsc.sc().getExecutorMemoryStatus()
    executor_memory_status_dict = sc._jvm.scala.collection.JavaConverters.mapAsJavaMapConverter(executor_memory_status).asJava()
    total_used_memory = 0
    for executor, values in executor_memory_status_dict.items():
        total_memory = values._1() / (1024 * 1024)  # Convert bytes to MB
        free_memory = values._2() / (1024 * 1024)    # Convert bytes to MB
        used_memory = total_memory - free_memory
        total_used_memory += used_memory
    return total_used_memory


def full_pipeline(df: DataFrame, sc, optimize: bool = False):
    df = preprocess_data(df, sc, optimize)
    model = fit_model(df, optimize)
    evaluate_model(model, df, optimize)


if __name__ == '__main__':
    args = parse_args()

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")

    spark = SparkSession.builder \
        .appName("Hadoop and Spark Lab") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    df = spark.read.csv("hdfs://namenode:9001/data.csv", header=True, inferSchema=True)

    if args.enable_optims:
        df = df.repartition(4)

    times = []
    RAMS = []
    for i in range(50):
        start = time.time()

        full_pipeline(df, sc, args.enable_optims)

        end = time.time()

        times.append(end - start)
        RAMS.append(get_executor_memory(sc))
        print(times[-1])
        print("Iter", i + 1)

    draw_stats(times, RAMS, args.data_nodes, args.enable_optims)

#df.show()
