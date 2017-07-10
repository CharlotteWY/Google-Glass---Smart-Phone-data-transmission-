from pyspark import SparkContext, SparkConf

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
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import time
import numpy as np

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


ranking_for_selected_brands = sc.textFile(ranking_for_selected_brands_folder_log).map(lambda line: line.split(mapping_separator)).map(lambda r: (int(r[0]), [int(r) for r in str(r[1]).replace("[", "").replace("]", "").split(", ")]))

data_valid = sc.textFile(data_valid_filtered_brands_customers_folder).map(lambda line: (int(line.split(mapping_separator)[0]), int(line.split(mapping_separator)[1]), float(line.split(mapping_separator)[2]))).filter(lambda p: p[1] in brand_ids)
data_valid_grouped = data_valid.groupBy(lambda line: line[0])
ranking_for_selected_brands_valid = data_valid_grouped.map(lambda r: (r[0], [p[1] for p in sorted([s for s in r[1]], key=lambda tup:tup[2])])) 

data_train = sc.textFile(data_train_filtered_brands_customers_folder).map(lambda line: (int(line.split(mapping_separator)[0]), int(line.split(mapping_separator)[1]), float(line.split(mapping_separator)[2]))).filter(lambda p: p[1] in brand_ids)
data_train_grouped = data_train.groupBy(lambda line: line[0])
ranking_for_selected_brands_train = data_train_grouped.map(lambda r: (r[0], [p[1] for p in sorted([s for s in r[1]], key=lambda tup:tup[2])])) 

# user_id / valid / rank
data_valid_real = ranking_for_selected_brands_valid.join(ranking_for_selected_brands).map(lambda line: (line[0],(line[1][0],line[1][1])))

# user_id/ hit / valid / train 
data_all = data_valid_real.join(ranking_for_selected_brands_train).map(lambda line: (line[0],line[1][0][0],line[1][0][1],line[1][1]))

def CheckNewHit(p, r2, r3, n):
	if (p in r2[:n] and not(p in r3)): 
		return 1
	else:
		return 0

def CheckRepeatHit(p, r2, r3, n):
	if (p in r2[:n] and p in r3): 
		return 1
	else:
		return 0

def CheckRepeat(p, r3):
	if (p in r3): 
		return 1
	else:
		return 0

def CheckNew(p, r3):
	if (p in r3): 
		return 0
	else:
		return 1

def NewHitRate(r1,r2,r3,n):
	if sum([CheckNew(p,r3) for p in r1]) == 0:
		return 2
	else: 
		return sum([CheckNewHit(p,r2,r3,n) for p in r1])/sum([CheckNew(p,r3) for p in r1])

def RepeatHitRate(r1,r2,r3,n):
	if sum([CheckRepeat(p,r3) for p in r1]) == 0:
		return 2
	else: 
		return sum([CheckRepeatHit(p,r2,r3,n) for p in r1])/sum([CheckRepeat(p,r3) for p in r1])

def AllHitRate(r1,r2,r3,n):
	if sum([CheckRepeat(p,r3) for p in r1]) == 0:
		return 2
	else: 
		return (sum([CheckRepeatHit(p,r2,r3,n) for p in r1])+sum([CheckNewHit(p,r2,r3,n) for p in r1]))/(sum([CheckRepeat(p,r3) for p in r1])+sum([CheckNew(p,r3) for p in r1]))

result = data_all.map(lambda r: (r[0], NewHitRate(r[1],r[2],r[3],10),RepeatHitRate(r[1],r[2],r[3],10)))
NewRate = np.mean(result.filter(lambda r: r[1]<1.5).map(lambda r: r[1]).collect())
RepeatRate = np.mean(result.filter(lambda r: r[2]<1.5).map(lambda r: r[2]).collect())

n = [1,3,5,8,10,12,15,20,30,50,100,200]
newRate_all = []
repeatRate_all = []
allRate_all = []
for q in n:
	result = data_all.map(lambda r: (r[0], NewHitRate(r[1],r[2],r[3],q),RepeatHitRate(r[1],r[2],r[3],q),AllHitRate(r[1],r[2],r[3],q)))
	NewRate = np.mean(result.filter(lambda r: r[1]<1.5).map(lambda r: r[1]).collect())
	RepeatRate = np.mean(result.filter(lambda r: r[2]<1.5).map(lambda r: r[2]).collect())
	AllRate = np.mean(result.filter(lambda r: r[3]<1.5).map(lambda r: r[3]).collect())
	newRate_all.append(NewRate)
	repeatRate_all.append(RepeatRate)
	allRate_all.append(AllRate)

#num_allbrands_customers = count(result.filter(lambda r: r[3]<1.5).map(lambda r: r[3]).collect())
#num_repeatingbrands_customers = sum(result.filter(lambda r: r[2]<1.5).map(lambda r: r[2]).collect())
#num_newbrands_customers = sum(result.filter(lambda r: r[1]<1.5).map(lambda r: r[1]).collect())

print("NewRate: "+ str(newRate_all)+ "RepeatRate: "+ str(repeatRate_all) + "AllRate: " + str(allRate_all))
	#"allbrands_customers:" + str(num_allbrands_customers) + "repeatingbrands_customers:" + str(num_repeatingbrands_customers) + "newbrands_customers:" + str(num_newbrands_customers))
