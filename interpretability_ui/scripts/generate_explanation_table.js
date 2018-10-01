


function GenerateExplanationTable()
{
	var num_dataset_images = $("#num_explanations_in_table").val();

	var selected_dataset = GetActiveDatasetName();
	
	GetDatasetExplanationTableImages(selected_dataset,num_dataset_images);
	//get images from dataset

//for dataset_image
	//for model
		//for explanation
			//create table


//for dataset_image
	//for model
		//for explanation
			//make call and populate table

	// var explanation_result_table_html = `<table id="image_explantion_table__test_img_1" class="table table-bordered image_explantion_table">
	// 						  <thead>
	// 						    <tr>
	// 						      <th scope="col"> 
	// 						      	<img id="explanation-table-input-image__test_img_1" class="explanation-table-input-image" src="test_content/testset_1_preview.jpg">
	// 						      </th>
	// 						      <th class="explanation_header" id="explanation_header__test_img_1__lime" scope="col">LIME</th>
	// 						      <th class="explanation_header" id="explanation_header__test_img_1__shap" scope="col">Shapley</th>
	// 						      <th class="explanation_header" id="explanation_header__test_img_1__influence_functions"scope="col">Influence Functions</th>
	// 						      <th class="explanation_header" id="explanation_header__test_img_1__lrp"scope="col">LRP</th>
	// 						    </tr>
	// 						  </thead>
	// 						  <tbody id="explanation-table-body__test_img_1" class="explanation-table-body">
	// 						    <tr>
	// 						      <th class="explanation_row_header" id="explanation_row_header__test_img_1__vgg16_imagenet" scope="row">VGG16</th>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg16_imagenet__lime" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg16_imagenet__lime" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg16_imagenet__shap" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg16_imagenet__shap" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg16_imagenet__influence_functions" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg16_imagenet__influence_functions" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg16_imagenet__lrp" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg16_imagenet__lrp" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
	// 						    </tr>

	// 						    <tr>
	// 						      <th class="explanation_row_header" id="explanation_row_header__test_img_1__vgg19_imagenet" scope="row">VGG19</th>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg19_imagenet__lime" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg19_imagenet__lime" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg19_imagenet__shap" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg19_imagenet__shap" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg19_imagenet__influence_functions" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg19_imagenet__influence_functions" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
							      
	// 						      <td class="explanation_result" id="explanation_result__test_img_1__vgg19_imagenet__lrp" >
	// 						      	<img id="explanation-table-result-image__test_img_1__vgg19_imagenet__lrp" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
	// 						      </td>
	// 						    </tr>
							    
	// 						   </tbody>
	// 						</table>`;

	// $("#card-deck__explanation-table").html($("#card-deck__explanation-table").html() + explanation_result_table_html);
}




function GetDatasetExplanationTableImages(dataset_name,num_inputs)
{

	browse_dataset_url = api_base_url + `/dataset-test-images?dataset=${dataset_name}&num_images=${num_inputs}`;
	browse_dataset_url = browse_dataset_url.split(" ").join("%20");  
	$.ajax({
	        url: browse_dataset_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	CreateAndPopulateExplanationTable(data,dataset_name);          
	        }
	    })
}


function CreateAndPopulateExplanationTable(input_images,dataset_name)
{
	var selected_models = GetActiveModels();
	var selected_explanations = GetActiveExplanations();



	for (var img_i = 0; img_i < input_images.length; img_i++){
		input_image = input_images[img_i];

		var image_identifier = input_image.image_name.replace(".jpg","");

		var explanation_result_table_html = `<table id="image_explantion_table__${image_identifier}" class="table table-bordered image_explantion_table">
							  <thead>
							    <tr>
							      <th scope="col"> 
							      	<img id="explanation-table-input-image__${image_identifier}" class="explanation-table-input-image" src="data:image/jpg;base64,${input_image.input}">
							      	<span class="explanation-table-input-image-ground-truth" id="explanation-table-input-image-ground-truth__${image_identifier}">${input_image.ground_truth}</span>
							      </th>`;

		for (var explanation_i = 0; explanation_i < selected_explanations.length; explanation_i++){
				current_explanation = selected_explanations[explanation_i];
					explanation_result_table_html = explanation_result_table_html + `<th class="explanation_header" id="explanation_header__${image_identifier}__${current_explanation.class_name}" scope="col">${current_explanation.explanation_name}</th>`;
			}


		explanation_result_table_html = explanation_result_table_html +`</tr>
							  </thead>
							  <tbody id="explanation-table-body__${image_identifier}" class="explanation-table-body">`;	

		//create rows							  
		for (var model_i = 0; model_i < selected_models.length; model_i++){
			current_model = selected_models[model_i];
			
			explanation_result_table_html = explanation_result_table_html +`<tr>
							      <th class="explanation_row_header" id="explanation_row_header__${image_identifier}__${current_model.class_name}" scope="row">${current_model.model_name}</th>`;
							      
							     
			//create cells
			for (var explanation_i = 0; explanation_i < selected_explanations.length; explanation_i++){
				current_explanation = selected_explanations[explanation_i];
				 	explanation_result_table_html = explanation_result_table_html +`<td class="explanation_result" id="explanation_result__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}" >
						      	<img id="explanation-table-result-image__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
						      	<span class="explanation-table-result-image-prediction" id="explanation-table-result-image-prediction__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}">"Fetching..."</span>
							    <span class="explanation-table-result-json-storage json-storage" id="explanation-table-result-json-storage__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}"></span>
							    
							     
						      </td>`;
			}

			explanation_result_table_html = explanation_result_table_html +`</tr>`;
		}



		explanation_result_table_html = explanation_result_table_html +`</tbody>
							</table>`;

		$("#card-deck__explanation-table").html($("#card-deck__explanation-table").html() + explanation_result_table_html);

		// for (var model_i = 0; model_i < selected_models.length; model_i++){
		// 	current_model = selected_models[model_i];
		// 	for (var explanation_i = 0; explanation_i < selected_explanations.length; explanation_i++){
		// 		current_explanation = selected_explanations[explanation_i];
		
		// 		PopulateExplanationTable(dataset_name,input_image,current_model,current_explanation);
		// 	}
		// }
	}


	PopulateExplanationTableSynchronously(dataset_name,input_images,selected_models,selected_explanations);


}


function PopulateExplanationTable(dataset_name,input_image,model,explanation)
{
	GetExplanationForTable(dataset_name,model,explanation,input_image.image_name,false,AddExplanationResultToTable);
}





function AddExplanationResultToTable(explanation_result,image_identifier,model_identifier,explanation_identifier)
{	
	var image_element_id = `explanation-table-result-image__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+image_element_id).attr("src","data:image/jpg;base64,"+explanation_result["explanation_image"]);

	var ground_truth_id = `explanation-table-result-image-prediction__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+ground_truth_id).html(explanation_result["prediction"]);


	var json_id = `explanation-table-result-json-storage__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+json_id).html(JSON.stringify(explanation_result));

}


function GetExplanationForTable(dataset_name,model,explanation,image_name,attribution_map,callback_function)
{
	var image_identifier = input_image.image_name.replace(".jpg","");
	var model_identifier = model.class_name;
	var explanation_identifier = explanation.class_name;

	var explain_url = api_base_url + `/explanation-explain?dataset=${dataset_name}&model=${model.model_name}&image=${image_name}&explanation=${explanation.explanation_name}&attribution_map=${attribution_map}`;

	$.ajax({
	        url: explain_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	callback_function(data,image_identifier,model_identifier,explanation_identifier);          
	        }
	    })
}





function PopulateExplanationTableSynchronously(dataset_name,input_images,selected_models,selected_explanations)
{
	GetExplanationForTableAndFetchNext(dataset_name,false,AddExplanationResultToTableAndFetchNext,input_images,selected_models,selected_explanations,0,0,0);
}

function AddExplanationResultToTableAndFetchNext(explanation_result,image_identifier,model_identifier,explanation_identifier,dataset_name,input_images,selected_models,selected_explanations,image_i,model_i,explanation_i)
{	
	var image_element_id = `explanation-table-result-image__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+image_element_id).attr("src","data:image/jpg;base64,"+explanation_result["explanation_image"]);

	var ground_truth_id = `explanation-table-result-image-prediction__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+ground_truth_id).html(explanation_result["prediction"]);


	var json_id = `explanation-table-result-json-storage__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+json_id).html(JSON.stringify(explanation_result));

	var counters = IncrementCounters(input_images,selected_models,selected_explanations,image_i,model_i,explanation_i);

	image_i = counters[2];
	model_i = counters[1];
	explanation_i = counters[0];

	if(image_i != -1)
	{
		GetExplanationForTableAndFetchNext(dataset_name,false,AddExplanationResultToTableAndFetchNext,input_images,selected_models,selected_explanations,image_i,model_i,explanation_i);	
	}
}

function IncrementCounters(input_images,selected_models,selected_explanations,image_i,model_i,explanation_i)
{
	explanation_i++;

	if(explanation_i >= selected_explanations.length)
	{
		explanation_i = 0;

		model_i++;

		if(model_i >= selected_models.length)
		{
			model_i = 0;

			image_i++;

			if(image_i >= input_images.length)
			{
				image_i = -1;
			}
		}	
	}

	return [explanation_i, model_i, image_i];
}

function GetExplanationForTableAndFetchNext(dataset_name,attribution_map,callback_function,input_images,selected_models,selected_explanations,image_i,model_i,explanation_i)
{	
	var input_image = input_images[image_i];
	var image_name = input_image.image_name;
	var model = selected_models[model_i];
	var explanation = selected_explanations[explanation_i];

	var image_identifier = input_image.image_name.replace(".jpg","");
	var model_identifier = model.class_name;
	var explanation_identifier = explanation.class_name;

	var explain_url = api_base_url + `/explanation-explain?dataset=${dataset_name}&model=${model.model_name}&image=${image_name}&explanation=${explanation.explanation_name}&attribution_map=${attribution_map}`;

	$.ajax({
	        url: explain_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	callback_function(data,image_identifier,model_identifier,explanation_identifier,dataset_name,input_images,selected_models,selected_explanations,image_i,model_i,explanation_i);          
	        }
	    })
}





function ProduceJsonForStorage()
{
	var explanation_results = $(".explanation_result");

	var json_strings = [];

	for (var result_i = 0; result_i < explanation_results.length; result_i++){
		var current_result = explanation_results[result_i];
		var result_id = current_result.id.replace("explanation_result__","");
		var result_identifiers = result_id.split("__");

		var result_dict = {};

		result_dict["image_name"] = result_identifiers[0];
		result_dict["model_class_name"] = result_identifiers[1];
		result_dict["explanation_class_name"] = result_identifiers[2];
		result_dict["result_json"] = JSON.parse($("#explanation-table-result-json-storage__"+result_id).html());

		var json_string = JSON.stringify(result_dict);

		json_strings.push(json_string);




	}



	$("#results_string_output").html('{"results":['+json_strings.join(",<br>")+']}');
	

	
}