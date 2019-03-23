





for i in $(seq 3 22)
do  
	python dataset_name_tidy.py "svrt_problem_$i"

	python create_dataset_spreadsheet.py "svrt_problem_$i"

done