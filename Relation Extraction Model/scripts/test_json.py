import json

filename = "../data/tacred/dev.json"

with open(filename, 'r') as f:
	data = f.readline()
	print(data)
	# for jsonstr in f.readlines(): # 按行读取json文件，每行为一个字符串
	# 	data = json.loads(jsonstr) # 将字符串转化为列表
	# 	print("--------")
	# 	print(data)
	# 	print("--------")
	# 	HoldTime = (data[0],data[3],data[6]) #取出列表中指定单个元素，这里取出了第1、4、7个，生成一个新的字符串
