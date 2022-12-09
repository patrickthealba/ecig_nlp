from pyspark.sql import SparkSession
from pyspark.ml import Pipeline as sparkPipeline
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import os, time
from multiprocessing.pool import ThreadPool
from pyspark.sql.types import ArrayType,IntegerType,FloatType,StringType

def spark_start():
    os.environ['SPARK_HOME'] = 'c:/spark'
    os.environ['HADOOP_HOME']= "C:/hadoop"
    os.environ['PYSPARK_PYTHON']=os.environ['CONDA_DEFAULT_ENV']+"/python.exe"
    os.environ['PYSPARK_DRIVER_PYTHON'] = os.environ['CONDA_DEFAULT_ENV']+"/python.exe"
    apname = str(time.time())
    spark = SparkSession.builder.appName(apname).master("local[128]")\
        .config("spark.jars","spark-nlp-assembly-4.0.2.jar,\
            spark-mssql-connector_2.12-1.2.0.jar,\
            mssql-jdbc-8.4.1.jre8.jar,\
            mssql-jdbc_auth.zip,\
            spark-nlp-jsl-4.0.2.jar")\
        .config("spark.driver.extraLibraryPath","./")\
        .config("spark.executor.extraLibraryPath","./")\
        .config("spark.driver.extraClassPath","./")\
        .config("spark.executor.extraClassPath","./")\
        .config("spark.driver.memory","500G")\
        .config("spark.executor.memory","500G")\
        .config("spark.sql.execution.arrow.pyspark.enabled","true")\
        .config("spark.task.maxFailures",'100') \
        .config("spark.executor.heartbeatInterval", "4000s") \
        .config("spark.network.timeout", "4100s") \
        .getOrCreate()
    return spark


def regex_filter(batch):
    r_min, r_max = batch
    s = time.time()
    
    print(f'     processing document {r_min} to document {r_max}      ')
    
    documentAssembler = DocumentAssembler() \
        .setInputCol('ReportText') \
        .setOutputCol("document")

    regex_matcher_doc = RegexMatcher() \
        .setInputCols('document') \
        .setOutputCol('reg_matches') \
        .setExternalRules(path='./ecg_term.txt', delimiter=',') \
        .setStrategy("MATCH_ALL") \
        .setLazyAnnotator(False)

    doc_pipeline = sparkPipeline(stages=[
        documentAssembler,regex_matcher_doc
    ])
    
    output_table = "[temp].[2022_docs_regex_filltered]"
    empty_df = spark.createDataFrame([['']]).toDF("ReportText")
    doc_Model = doc_pipeline.fit(empty_df)
    
    input_table = f'''(SELECT distinct A.tiudocumentsid, rowid,  lower(B.ReportText) as ReportText
                        FROM [nlp].[2022_complete_docs] A  
                        JOIN [src].TIUDocument B with(nolock)  
                        ON A.tiudocumentsid = B.tiudocumentsid  
                        WHERE rowid > = {r_min} AND rowid  < {r_max}) sub'''
                        
    df_ecig = spark.read.format("com.microsoft.sqlserver.jdbc.spark") \
            .option("url", url) \
            .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")\
            .option('dbtable', input_table)\
            .option('numPartitions', 128)\
            .option('partitionColumn', 'rowid')\
            .option('lowerBound', r_min)\
            .option('upperBound', r_max)\
            .option("schemaCheckEnabled", 'false')\
            .load().cache()
            
    doc_df = doc_Model.transform(df_ecig).filter(F.size(F.col('reg_matches')) > 0).cache()
    if doc_df.count()==0:
        return
    
    doc_df.select('TIUDocumentSID')\
        .write\
        .format("com.microsoft.sqlserver.jdbc.spark")\
        .mode("append") \
        .option("url", url) \
        .option("dbtable", output_table)\
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")\
        .save()
 
    with open('regex_filter.log', 'a') as f:
       f.write(str(r_min)+'\t'+str(r_max)+'\n')
       
    print(f'processed {r_min} to {r_max} in {time.time() - s :.0f} seconds')
    return


if __name__ == '__main__':
    startt = time.time()
    spark = spark_start()
    url = "jdbc:sqlserver://va.gov;integratedSecurity=true;"
    batches = [(i, i + 10000) for i in range(1, 35000000, 10000)]
    
    with open('regex_filter.log') as f:
       batch_start = [int(x.split()[0]) for x in f.readlines()]
       batches = [x for x in batches if x[0] not in batch_start]
       
    with ThreadPool(4) as p:
        p.map(regex_filter, batches)

    print(f'The task processed in {time.time()-startt:.0f} seconds')




