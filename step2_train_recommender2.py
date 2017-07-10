from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setAppName('step1.3')
conf.set("spark.driver.memory", "5G")
conf.set("spark.executor.instances", "4")
conf.set("spark.executor.memory", "20G")
conf.set("spark.executor.cores", "4")
                       
sc = SparkContext(conf=conf) 

from math import *
from config import *
from helper import *
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import time

t1 = time.time()

# custID, brandID, txn
train_file = sc.textFile(data_train_filtered_brands_customers_folder)
# convert to Rating object
ratings = train_file.map(lambda line: line.split(mapping_separator)).map(lambda r: Rating(int(r[0]), int(r[1]), log(1+ float(r[2])*10000)))
ratings.cache()
ratings.first()

# Build the recommendation model using ALS
model = ALS.trainImplicit(ratings=ratings, rank=cf_rank, iterations=cf_numIterations, lambda_=cf_lambda, alpha=cf_alpha)

# Save and load model
hadoop_file_delete(cf_model_folder_log)
model.save(sc, cf_model_folder_log)

t2 = time.time()
print("Time elapsed: "+ str(t2-t1)+" seconds")
print("Done!")