# Bibliotecas
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

def criar_sessao_spark():
    return SparkSession.builder.appName("Treinamento").getOrCreate()

def leitura_csv(spark, caminho):

    schema = StructType([
        StructField("datetime", TimestampType(), True),
        StructField("instance_type", StringType(), True),
        StructField("os", StringType(), True),
        StructField("region", StringType(), True),
        StructField("price", DoubleType(), True)
    ])

    return spark.read.csv(caminho, header = False, schema = schema)

def quebra_string_tipo(df):
    return df.withColumn("tipo", F.split(df['instance_type'], "\.")[0])

def calcula_media_precos_familia(df):
    return df.groupBy('tipo').agg(F.avg(df.price).alias('Avg')).sort(F.col('Avg').desc())

def join_media_preco(df, df_type):
    return df.alias('A').join(df_type.alias('B'),
                              F.col('A.tipo') == F.col('B.tipo'),
                              how = 'left').select(
                                  [F.col('A.' + xx) for xx in df.columns] + [F.col('B.avg')]
                              )

def classifica_preco(df):
    return df.withColumn('classificacao', F.when(df.price > df.avg, "ALTO").otherwise("BAIXO"))

def escreve_arquivo_parquet(df, caminho):
    df.write.mode('overwrite').parquet(caminho)

if __name__ == '__main__':

    spark = criar_sessao_spark()
    df = leitura_csv(spark, 'datasets/*.csv')
    df = quebra_string_tipo(df.limit(1000))
    df_type = calcula_media_precos_familia(df)
    df = join_media_preco(df, df_type)
    df = classifica_preco(df)
    escreve_arquivo_parquet(df, 'datasets/treinamento.parquet')
    
    
    
    


    