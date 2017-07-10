#///////////////////////////////////////////
#// this script retrieves list of brands that have low #TXN
#//////////////////////////////////////////

from pyspark import SparkContext, SparkConf
sc = SparkContext()

from config import *
from helper import *

# copy this file manually to output_path!!! # brand_list_filename 

print("loading data...")

# load the selected brands
brand_list = sc.textFile(brand_list_filename).filter(lambda r: r!="").map(lambda r:r.split(mapping_separator)[1])
brandName_brandID = sc.textFile(brandName_brandID_folder).map(lambda line:(line.split(mapping_separator)[0], line.split(mapping_separator)[1]))
brand_id_list = brand_list.map(lambda r:(r,1)).join(brandName_brandID).map(lambda r: int(r[1][1]))

# load original data
data_train_file = sc.textFile(data_train_folder) # (cust_id, brand_id, txn)
data_valid_file = sc.textFile(data_valid_folder) # (cust_id, brand_id, txn)

# calculate # transactions per brand
brands_txn = data_train_file.map(lambda line: line.split(mapping_separator)).map(lambda line: (int(line[1]), float(line[2]))\
).reduceByKey(lambda x,y: x+y) # (brand_id, txn)

# get brands after filtering
brands_txn_thresholded = brands_txn.filter(lambda a: a[1] > brand_filtering_threshold)

# save to file
hadoop_file_delete(topk_brands_txn_all_folder)
brands_txn.map(lambda r: str(r[0]) + mapping_separator+str(r[1])).coalesce(1,True).saveAsTextFile(topk_brands_txn_all_folder)

print("Filter the training and validation data set:")
brands_selected = brands_txn_thresholded.map(lambda r: r[0]).union(brand_id_list).zipWithUniqueId().collectAsMap()

# filter the train data
data_train_filtered_brands = data_train_file.filter(lambda line: int(line.split(mapping_separator)[1]) in brands_selected)
hadoop_file_delete(data_train_filtered_brands_folder)
data_train_filtered_brands.saveAsTextFile(data_train_filtered_brands_folder)
#data_train_filtered_brands.coalesce(1,true).saveAsTextFile(data_train_filtered_brands_folder+"_one_file")

# filter the validation data
data_valid_filtered_brands = data_valid_file.filter(lambda line: int(line.split(mapping_separator)[1]) in brands_selected)
hadoop_file_delete(data_valid_filtered_brands_folder)
data_valid_filtered_brands.saveAsTextFile(data_valid_filtered_brands_folder)
# data_valid_filtered_brands.coalesce(1,true).saveAsTextFile(data_valid_filtered_brands_folder+"_one_file")

print("All brands: " + str(brands_txn.count()) + ". Selected: " + str(len(brands_selected)))
print("Done!")