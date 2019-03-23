

csv_name = "svrt_problem_{0}.csv"

train_csv_name = "svrt_problem_{0}_train_split.csv"

split_csv_name = "svrt_problem_{0}_split.csv"

file_names = [csv_name,split_csv_name,train_csv_name]


for i in range(1,24):
	for file_name in file_names:
		file_string = ""
		with open(file_name.format(i),"r") as f:
			file_string = f.read()

		with open(file_name.format(i),"w") as f:
			f.write(file_string.replace(".png",".jpg"))
		