from pyspark.sql import SparkSession
from pyspark.ml import Pipeline as sparkPipeline
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import os, time
from multiprocessing.pool import ThreadPool
from pyspark.sql.types import ArrayType,IntegerType,FloatType,StringType
from transformers import (TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification)
import torch
from transformers.utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from transformers.pipelines.base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline
import numpy as np
from onnxruntime import (InferenceSession, SessionOptions, GraphOptimizationLevel)

if is_tf_available():
    from transformers.models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

if is_torch_available():
    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


def start_onnx_session(onnx_model_file_path):
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession('ecig_classification.onnx',
                               sess_options=options,
                               providers=['CPUExecutionProvider'])
    session.disable_fallback()
    return session


def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))

def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    """,
)
class OnnxTextClassificationPipeline(TextClassificationPipeline):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)

    def _forward(self, model_inputs):
        inputs = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = session.run(None, input_feed=inputs)[0]  #Changed to onnxruntime optimization
        logits = torch.tensor(logits)
        return {
            "logits": logits,
            #             "special_tokens_mask": special_tokens_mask,
            # "offset_mapping": offset_mapping,
            #             "sentence":  sentence,
            **model_inputs,
        }

    def preprocess(self, sentence):
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=truncation,
        )

        return model_inputs

    def postprocess(self, model_outputs, function_to_apply=None, return_all_scores=False):
        # Default value before `set_parameters`
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE

        outputs = model_outputs["logits"]
        outputs = outputs.numpy()
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")
        if return_all_scores:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores[0]}
        else:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}


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


def read_save(batch):
    r_min, r_max = batch
    s = time.time()
    
    print(f'     processing document {r_min} to document {r_max}      ')
    
    documentAssembler = DocumentAssembler() \
        .setInputCol('Text') \
        .setOutputCol("document")

    sentencerDL = SentenceDetectorDLModel \
        .load(
        "./pretrained_models/sentence_detector_dl_healthcare_en_3.2.0_3.0_1628678815210") \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    regex_matcher_doc = RegexMatcher() \
        .setInputCols('sentence') \
        .setOutputCol('reg_matches') \
        .setExternalRules(path='./ecg_term.txt', delimiter=',') \
        .setStrategy("MATCH_ALL") \
        .setLazyAnnotator(False)

    doc_pipeline = sparkPipeline(stages=[
        documentAssembler,sentencerDL,regex_matcher_doc
    ])
    
    output_table = "[nlp].[2022_predictions]"
    empty_df = spark.createDataFrame([['']]).toDF("Text")
    doc_Model = doc_pipeline.fit(empty_df)
    
    input_table = f'''(SELECT distinct A.documentsid, rowid,  lower(B.Text) as Text
                        FROM [temp].[2022_docs_regex_filltered] A  
                        JOIN [src].TIUDocument B with(nolock)  
                        ON A.documentsid = B.documentsid  
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
            
    doc_df = doc_Model.transform(df_ecig).cache()

    context_df = doc_df.select('documentsid', 'rowid',
                         'sentence', F.explode('reg_matches').alias('match')) \
        .select('documentsid', 'match', 'rowid',
                F.explode('sentence').alias('sent')) \
        .select('documentsid', 'rowid',
                F.col('sent')['result'].alias('original_sentence'),
                F.col('sent')['metadata']['sentence'].alias('sent_num'),
                F.col('match')['metadata']['sentence'].alias("matched_sent"),
                F.col('match')['begin'].alias("spanStart"),
                F.col('match')['end'].alias("spanEnd"),
                F.trim(F.lower(F.col('match')['result'])).alias('matched_term')
                ) \
        .filter("matched_sent-sent_num<=1 AND matched_sent-sent_num>=-1") \
        .withColumn("InstanceID",
                  F.concat(F.col('DocumentSID'), F.lit('_'), F.col('matched_sent')))\
        .dropDuplicates(['InstanceID','sent_num'])\
        .withColumn('original_sentence', F.regexp_replace('original_sentence',r'\r',''))\
        .orderBy('InstanceID', 'sent_num').cache()

    data = context_df.groupBy('InstanceID') \
        .agg(F.first('DocumentSID').alias('DocumentSID'),
             F.first('spanStart').alias('spanStart'),
             F.first('spanEnd').alias('spanEnd'),
             F.concat_ws(" ", F.collect_list('ORIGINAL_SENTENCE')).alias('text'),
             F.concat_ws(' ', F.collect_set('matched_term')).alias('term')
             ).cache()
             
    context = data.toPandas()
    
    print(f'Start predicting {context.shape[0]} instances')
    
    ps = time.time()
    predictions = onnx_pipeline(context['text'].to_list())
    
    print(f'      predict {context.shape[0]} instances in {time.time() - ps:.0f} seconds      ')
        
    context['prediction'] = [x['label'] for x in predictions]
    context[['prob_active-user', 'prob_usage-unknown', 'prob_non-user', 'prob_irrelevant', 'prob_former-user']] = [p['score'] for p in predictions]
    
    spark.createDataFrame(context)\
        .select('rowid', 'InstanceID', 'DocumentSID', 'spanStart', 'spanEnd', 'term', 'prediction',
                'prob_active-user', 'prob_usage-unknown', 'prob_non-user', 'prob_irrelevant', 'prob_former-user')\
        .write\
        .format("com.microsoft.sqlserver.jdbc.spark")\
        .mode("append") \
        .option("url", url) \
        .option("dbtable", output_table)\
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")\
        .save()

    print(f'   |||   processed {context.shape[0]} instances in {time.time() - s:.0f} seconds   |||   ')
    
    with open('batch_prediction.log', 'a') as f:
       f.write(str(r_min)+'\t'+str(r_max)+'\n')
       
    print(f'processed {r_min} to {r_max} in {time.time() - s :.0f} seconds')
    return



if __name__ == '__main__':
    startt = time.time()
    spark = spark_start()
    url = "jdbc:sqlserver://va.gov;integratedSecurity=true;"
    session = start_onnx_session('ecig_classification.onnx')
    model_name = 'final_model/'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    onnx_pipeline = OnnxTextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework='pt',
        return_all_scores=True
    )

    batches = [(i, i + 10000) for i in range(1, 35000000, 10000)]
    
    with open('batch_prediction.log') as f:
       batch_start = [int(x.split()[0]) for x in f.readlines()]
       batches = [x for x in batches if x[0] not in batch_start]
       
    with ThreadPool(2) as p:
        p.map(read_save, batches)

    print(f'The task processed in {time.time()-startt:.0f} seconds')




