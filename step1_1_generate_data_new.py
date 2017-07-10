# Obtain the brand mapping and customer mapping from input triplets
# Convert the triplet from (party_id, brand, #txn) to (cust_id, brand_id, #txn)

from pyspark import SparkContext, SparkConf
#sc.setLogLevel("ERROR")

conf = SparkConf()
conf.setAppName('step3_ranking')
conf.set("spark.driver.memory", "5G")
conf.set("spark.executor.instances", "4")
conf.set("spark.executor.memory", "20G")
conf.set("spark.executor.cores", "4")   
#conf.set("spark.yarn.executor.memoryOverhead","0.2")
sc = SparkContext(conf=conf)


#sc.addPyFile('config.py')
from config import *
from helper import *

train_file_visa = sc.textFile(raw_train_path).map(lambda line: line.split(train_separator)).map(lambda r: (r[0], r[2], float(r[3])))
valid_file_visa = sc.textFile(raw_valid_path).map(lambda line: line.split(train_separator)).map(lambda r: (r[0], r[2], float(r[3])))
train_file_nets = sc.textFile(raw_train_path_nets).map(lambda line: (str(line.split(train_separator)[0]), str(line.split(train_separator)[1]), str(line.split(train_separator)[2]))).filter(lambda r: r[0]!="party_id").map(lambda r: (r[0],r[1],float(r[2])))
valid_file_nets = sc.textFile(raw_valid_path_nets).map(lambda line: (str(line.split(train_separator)[0]), str(line.split(train_separator)[1]), str(line.split(train_separator)[2]))).filter(lambda r: r[0]!="party_id").map(lambda r: (r[0],r[1],float(r[2])))


# => ((partyID, brandName), sum of #TXN)
train_ratings = train_file_visa.union(train_file_nets).map(lambda line: ((line[0], line[1]),float(line[2]))).reduceByKey(lambda x, y: x + y)
valid_ratings = valid_file_visa.union(valid_file_nets).map(lambda line: ((line[0], line[1]),float(line[2]))).reduceByKey(lambda x, y: x + y)
#//////////// Columns in data files are: PartyID,MCC,BrandName,#TXN

# create brand mapping => distinct Brands => (brandName, brandID)
brandName_brandID = train_file_visa.map(lambda line: line[1]).distinct().zipWithIndex().map(lambda line: (line[0], line[1] + 1))
brandName_brandID.cache()

# create customer mapping => distinct PartyID => (partyID, custID)
partyID_custID = train_file_visa.map(lambda line: line[0]).distinct().zipWithIndex().map(lambda line: (line[0], line[1] + 1))
partyID_custID.cache()

num_brand = brandName_brandID.count()

# => (PartyID, (brandName, sum of #TXN)) => (PartyID, ((brandName, sum of #TXN), custID)) => (brandName, (custID, sum of #TXN))
train_ratings_custid = train_ratings.map(lambda line: (line[0][0], (line[0][1], line[1]))
).join(partyID_custID
).map(lambda line: (line[1][0][0], (line[1][1], line[1][0][1])))
	
# => (brandName, ((custID, sum of #TXN), brandID)) => (custID, brandID, sum of #TXN)
train_custID_brandID_txn = train_ratings_custid.join(brandName_brandID)\
.map(lambda line: str(line[1][0][0]) + mapping_separator + str(line[1][1]) + mapping_separator + str(line[1][0][1]))

#valid_ratings = valid_file.map(lambda line: ((line.split(valid_separator)[0], line.split(valid_separator)[2]),float(line.split(valid_separator)[3]))).reduceByKey(lambda x, y: x + y)

valid_ratings_custid = valid_ratings.map(lambda line: (line[0][0], (line[0][1], line[1])))\
.join(partyID_custID).map(lambda line: (line[1][0][0], (line[1][1], line[1][0][1])))
valid_custID_brandID_txn = valid_ratings_custid.join(brandName_brandID).map(
lambda line: str(line[1][0][0]) + mapping_separator + str(line[1][1]) + mapping_separator + str(line[1][0][1]))


# save reference tables and interaction matrix to hdfs
hadoop_file_delete(partyID_custID_folder)
hadoop_file_delete(brandName_brandID_folder)
hadoop_file_delete(data_train_folder)
hadoop_file_delete(data_valid_folder)
partyID_custID.map(lambda line: line[0] + mapping_separator + str(line[1])).saveAsTextFile(partyID_custID_folder)
brandName_brandID.map(lambda line: line[0] + mapping_separator + str(line[1])).saveAsTextFile(brandName_brandID_folder)
train_custID_brandID_txn.saveAsTextFile(data_train_folder)
valid_custID_brandID_txn.saveAsTextFile(data_valid_folder)