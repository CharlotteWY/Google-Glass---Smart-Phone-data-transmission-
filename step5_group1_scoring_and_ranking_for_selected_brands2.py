from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setAppName('step3_ranking')
conf.set("spark.driver.memory", "5G")
conf.set("spark.executor.instances", "4")
conf.set("spark.executor.memory", "20G")
conf.set("spark.executor.cores", "4")   
#conf.set("spark.yarn.executor.memoryOverhead","0.2")
sc = SparkContext(conf=conf)


#sc = SparkContext()

# function to calcuate ranking
import numpy as np
def ranked(uFeature, pFeatures):
    s = [(p[0], np.dot(uFeature[1],p[1])) for p in pFeatures]
    s.sort(key=lambda tup: -tup[1])
    return [r[0] for r in s]

from config import *
from helper import *
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import time

brand_list = sc.textFile(brand_list_filename).filter(lambda r: r!="").map(lambda r: r.split(mapping_separator)[1]).collect()

t1 = time.time()
print("loading data...")

# brandName_brandID: <brand_name, brand_id>, prefilter the requested brand list
brandName_brandID_file = sc.textFile(brandName_brandID_folder)
brandName_brandID = brandName_brandID_file.filter(lambda line: line.split(mapping_separator)[0] in brand_list
).map(lambda line: line.split(mapping_separator)
).filter(lambda line: len(line) == 2 # keep only those that can be splitted into two columns
).map(lambda line: (line[0], int(line[1])) # (brand_name, brand_id)
).cache()
# => (brand_name, brand_id)

brand_ids = brandName_brandID.map(lambda line: int(line[1])).collect() # => (brand_id)

model = MatrixFactorizationModel.load(sc, cf_model_folder_log)
model.productFeatures().cache()
model.userFeatures().cache()

users = model.userFeatures().map(lambda r: r[0])

################################################# Method 2: calculate manually
num_products = model.productFeatures().count()
selected_productFeatures = sc.broadcast(model.productFeatures().filter(lambda p: p[0] in brand_ids).collect())
userFeatures = model.userFeatures()

# calcuate manually the ranking from feature vectors
data = userFeatures.map(lambda u: (u[0], ranked(u, selected_productFeatures.value)))

hadoop_file_delete(ranking_for_selected_brands_folder_log)
data.map(lambda r: str(r[0]) + mapping_separator+str(r[1])).saveAsTextFile(ranking_for_selected_brands_folder_log)
print("calculation done")


################################################# Get Ranking 

data_valid = sc.textFile(data_valid_filtered_brands_customers_folder).map(lambda line: (int(line.split(mapping_separator)[0]), int(line.split(mapping_separator)[1]), float(line.split(mapping_separator)[2]))).filter(lambda p: p[1] in brand_ids)
data_valid_grouped = data_valid.groupBy(lambda line: line[0])

ranking_for_selected_brands_valid = data_valid_grouped.map(lambda r: (r[0], [p[1] for p in sorted([s for s in r[1]], key=lambda tup:tup[2])])) 
ranking_for_selected_brands_valid.cache()
hadoop_file_delete(ranking_for_selected_brands_valid_folder_log)
ranking_for_selected_brands_valid.saveAsTextFile(ranking_for_selected_brands_valid_folder_log)

data_all = ranking_for_selected_brands_valid.join(data).map(lambda line: (line[0],line[1][0],line[1][1]))

ranking = data_all.map(lambda r: (r[0], [r[2].index(a) for a in r[1]]))

#hadoop_file_delete(ranking_folder)
#ranking.saveAsTextFile(ranking_folder)

result = ranking.map(lambda r: (r[0], np.mean(r[1])/len(brand_ids)))
hadoop_file_delete(result_folder_log)
result.map(lambda r: str(r[0]) + mapping_separator + str(r[1])).saveAsTextFile(result_folder_log)

ranking_result = np.mean(result.map(lambda r: r[1]).collect())



#################################################Get Benchmark
def getRank(r2, a):
	if a in r2: 
		return r2.index(a)
	else:
		return len(brand_ids)/2

data_train = sc.textFile(data_train_filtered_brands_customers_folder).map(lambda line: (int(line.split(mapping_separator)[0]), int(line.split(mapping_separator)[1]), float(line.split(mapping_separator)[2])))
data_train_grouped = data_train.groupBy(lambda line: line[0])
ranking_for_selected_brands_train = data_train_grouped.map(lambda r: (r[0], [p[1] for p in sorted([s for s in r[1]], key=lambda tup:tup[2])])) 
ranking_for_selected_brands_train.cache()
hadoop_file_delete(ranking_for_selected_brands_train_folder_log)
ranking_for_selected_brands_train.saveAsTextFile(ranking_for_selected_brands_train_folder_log)
train_all = ranking_for_selected_brands_valid.join(ranking_for_selected_brands_train).map(lambda line: (line[0],line[1][0],line[1][1]))
ranking_train = train_all.map(lambda r: (r[0], [getRank(r[2],a) for a in r[1]]))
benchmark = ranking_train.map(lambda r: (r[0], np.mean(r[1])/len(brand_ids)))
hadoop_file_delete(benchmark_folder_log)
benchmark.map(lambda r: str(r[0]) + mapping_separator + str(r[1])).saveAsTextFile(benchmark_folder_log)

benchmark_result = np.mean(benchmark.map(lambda r: r[1]).collect())
print("Final average ranking: "+ str(ranking_result)+ "benchmark average ranking: "+ str(benchmark_result))
print("Done!")


