import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import count, when, isnull,col
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame


def fit_model(data: DataFrame, optimize: bool = False):
    kmeans = KMeans(k=7)
    kmeans.setSeed(1)
    kmeans.setMaxIter(10)

    model = kmeans.fit(data)

    return model


def evaluate_model(model: KMeans, df: DataFrame, optimize: bool = False):
    preds = model.transform(df)

    evaluator = ClusteringEvaluator(predictionCol='prediction',
                                    featuresCol='scaledFeatures',
                                    metricName='silhouette',
                                    distanceMeasure='squaredEuclidean')

    score = evaluator.evaluate(preds)

    print('Silhouette Score:', score)


