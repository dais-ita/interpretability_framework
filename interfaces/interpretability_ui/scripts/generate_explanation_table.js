function GenerateExplanationTable()
{
	var num_dataset_images = $("#num_explanations_in_table").val();

	var selected_dataset = GetActiveDatasetName();
	
	GetDatasetExplanationTableImages(selected_dataset,num_dataset_images);
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

	// var table_scaffold_json = {"table":{"input_images":input_images,"selected_models":selected_models,"selected_explanations":selected_explanations}};



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
							    <span class="explanation-table-result-json-storage json-storage" id="explanation-table-result-json-storage__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}">{}</span>
							    
							     
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
	GetExplanationForTable(dataset_name,model,explanation,input_image.image_name,false,AddExplanationResultToTable)
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
	    });
}




function ProduceJsonForStorage()
{
	$("#results_string_output").html("Generating JSON for download...");
	
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
	//$("#results_string_output").html('{"results":['+json_strings.join(",<br>")+']}');
	DownloadTableJson('{"explanation_table_data":{"results":['+json_strings.join(",\n")+']}, \n\n"table_html":'+JSON.stringify({"html":$("#card-deck__explanation-table").html()})+'}');
	
}


function SaveTableAsJson()
{
	var table_json = CollectJsonFromTable();

	alert("save");
}

function CollectJsonFromTable()
{
	var table_jsons = $(".explanation-table-result-json-storage").html();
	return table_jsons;
}

function DownloadTableJson(table_json_string)
{
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(table_json_string));
  element.setAttribute('download', "json_table.json");

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

function LoadFromFile()
{
	var files = document.getElementById("table_json_input").files;
	BuildTableFromFile(files[0]);
	
}

function BuildTableFromJsonString(table_json_string)
{
	var table_json = JSON.parse(table_json_string);

	alert("building table...");
	$("#card-deck__explanation-table").html(table_json.table_html.html);

	
}

function BuildTableFromFile(f)
{
    var reader = new FileReader();
    reader.onload = (function(file) {
        return function(e) {
          BuildTableFromJsonString(e.target.result);
        };
      })(f);

    reader.readAsText(f);
    
}

$('#table_json_input').on('change',function(){
    var files = this.files;

    $("#load_table_label").html(files[0].name);
})