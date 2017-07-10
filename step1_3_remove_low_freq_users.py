#////////////////////////////////////////////
#// this script retrieves list of customers that have low #TXN
#//////////////////////////////////////////

from pyspark import SparkContext

conf = SparkConf()
conf.setAppName('step3_ranking')
conf.set("spark.driver.memory", "5G")
conf.set("spark.executor.instances", "4")
conf.set("spark.executor.memory", "20G")
conf.set("spark.executor.cores", "4")   
#conf.set("spark.yarn.executor.memoryOverhead","0.2")
sc = SparkContext(conf=conf)

from config import *
from helper import *

print("loading data...")

# load brand-filtered data
data_train_filtered_brands_file = sc.textFile(data_train_filtered_brands_folder) # (cust_id, brand_id, txn)
data_valid_filtered_brands_file = sc.textFile(data_valid_filtered_brands_folder) # (cust_id, brand_id, txn)

# get # brands each customer transacted
customers_numbrands = data_train_filtered_brands_file.map(lambda line: line.split(mapping_separator)\
).map(lambda line: (int(line[0]), 1) # change to 'line => (line(0).toInt, line(2).toDouble)' for cust_id, txn
).reduceByKey(lambda x,y:x+y) # cust_id, numbrand 

# customer filtering
customers_numbrands_selected = customers_numbrands.filter(lambda a: a[1] > customer_filtering_threshold)
customers_numbrands_excluded = customers_numbrands.filter(lambda a: a[1] <= customer_filtering_threshold)

# save customer lists
hadoop_file_delete(topk_customers_txn_all_folder)
customers_numbrands.map(lambda r: str(r[0]) + mapping_separator + str(r[1])).coalesce(1,True).saveAsTextFile(topk_customers_txn_all_folder)
hadoop_file_delete(topk_customers_txn_selected_folder)
customers_numbrands_selected.map(lambda r: str(r[0]) + mapping_separator + str(r[1])).coalesce(1,True).saveAsTextFile(topk_customers_txn_selected_folder)

customers_selected = customers_numbrands_selected.map(lambda r: r[0]).zipWithUniqueId().collectAsMap()
# filter the train data
data_train_filtered_brands_customers = data_train_filtered_brands_file.filter(lambda line: int(line.split(mapping_separator)[0]) in customers_selected)
hadoop_file_delete(data_train_filtered_brands_customers_folder)
data_train_filtered_brands_customers.saveAsTextFile(data_train_filtered_brands_customers_folder)

# filter the validation data
data_valid_filtered_brands_customers = data_valid_filtered_brands_file.filter(lambda line: int(line.split(mapping_separator)[0]) in customers_selected)
hadoop_file_delete(data_valid_filtered_brands_customers_folder)
data_valid_filtered_brands_customers.saveAsTextFile(data_valid_filtered_brands_customers_folder)

print("All customers: " + str(customers_numbrands.count()) + ". Selected: " + str(customers_numbrands_selected.count()) + ". Excluded: " + str(customers_numbrands_excluded.count())+ "data_train_count: " + str(data_train_filtered_brands_customers.count())+ "data_valid_count: " + str(data_valid_filtered_brands_customers.count()))
print("data_train_count: " + str(data_train_filtered_brands_customers.count()))
print("data_valid_count: " + str(data_valid_filtered_brands_customers.count()))
print("Done!")
