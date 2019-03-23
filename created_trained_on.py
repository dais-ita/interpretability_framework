


accuracies = [
0.86328125,
1,
1,
1,
0.96875,
0.8828125,
0.5,
1,
0.98828125,
0.9921875,
1,
1,
0.9765625,
0.93359375,
0.97265625,
0.97265625,
0.93359375,
0.95703125,
0.859375,
0.58984375,
0.765625,
0.9609375,
0.9921875,
]

traind_ons = []

for i in range(23):
	accuracy = accuracies[i]
	p_num = i+1

	train_on_dict = '''	
	{
		"dataset_name":"SVRT Problem '''+str(p_num)+'''",
		"model_path":"saved_models/svrt_problem_'''+str(p_num)+'''_",
		"training_allocation_path":"dataset_csvs/svrt_problem_'''+str(p_num)+'''_train_split.csv",
		"test_accuracy":'''+str(accuracy)+''',
		"training_time":0,
		"training_steps":40
	}'''

	traind_ons.append(str(train_on_dict))


print(",\n".join(traind_ons))