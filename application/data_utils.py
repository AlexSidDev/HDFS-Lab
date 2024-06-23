import pandas as pd
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import count, when, isnull, col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import DataFrame


def encode_categorical(df: DataFrame, sc, optimize: bool = False):
    categorical_columns = [col[0] for col in df.dtypes if col[1] == 'string']

    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]

    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=column + "_encoded") for indexer, column in zip(indexers, categorical_columns)]

    pipeline = Pipeline(stages=indexers + encoders)

    model = pipeline.fit(df)

    df = model.transform(df)

    for column in categorical_columns:
        df = df.drop(column)
        df = df.drop(column + "_index")

    return df


def preprocess_data(df: DataFrame, sc, optimize: bool = False):

    df = df.drop('Serial Number', 'Date Recorded', 'Town', 'Address')
    pd_df = df.pandas_api()
    pd_df = pd_df.drop(axis=1, columns=pd.Series(pd_df.columns)[(pd_df.isna().sum() > len(pd_df.index) / 2).values])
    pd_df.dropna(inplace=True)

    df = pd_df.to_spark()

    df = encode_categorical(df, sc, optimize)

    vec_assembler = VectorAssembler(inputCols=df.columns,
                                    outputCol='features')

    final_data = vec_assembler.transform(df)

    scaler = StandardScaler(inputCol="features",
                            outputCol="scaledFeatures",
                            withStd=True,
                            withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(final_data)

    # Normalize each feature to have unit standard deviation.
    final_data = scalerModel.transform(final_data)

    return final_data
