base_data_api_url = "http://localhost:6501/";
base_data_api_url = "http://services.futurelabsolutions.com:6080/";
	

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


	CreateTable(input_images,dataset_name,selected_models,selected_explanations);	

	PopulateExplanationTableSynchronously(dataset_name,input_images,selected_models,selected_explanations);


}

function CreateTable(input_images,dataset_name,selected_models,selected_explanations)
{
	var dataset_identifier = dataset_name.split(" ").join("_").replace("(","").replace(")","");
	
	$("#card-deck__explanation-table").html('<span id="explanation_table_json_storage"></span>');
	$("#explanation_table_json_storage").html(JSON.stringify({"input_images":input_images,"dataset_name":dataset_name,"selected_models":selected_models,"selected_explanations":selected_explanations}))
		
	for (var img_i = 0; img_i < input_images.length; img_i++){
		input_image = input_images[img_i];

		var image_identifier = input_image.image_name.replace(".jpg","").replace(".JPEG","");

		var explanation_result_table_html = `
							<table id="image_explantion_table__${dataset_identifier}__${image_identifier}" class="table table-bordered image_explantion_table">
							  <thead>
							    <tr>
							      <th scope="col"> 
							      	<img id="explanation-table-input-image__${dataset_identifier}__${image_identifier}" class="explanation-table-input-image" src="data:image/jpg;base64,${input_image.input}">
							      	<span class="explanation-table-input-image-ground-truth" id="explanation-table-input-image-ground-truth__${dataset_identifier}__${image_identifier}">${img_i+1}) ${input_image["image_name"]} - ${input_image.ground_truth}</span>
							      </th>`;

		for (var explanation_i = 0; explanation_i < selected_explanations.length; explanation_i++){
				current_explanation = selected_explanations[explanation_i];
					explanation_result_table_html = explanation_result_table_html + `<th class="explanation_header" id="explanation_header__${dataset_identifier}__${image_identifier}__${current_explanation.class_name}" scope="col">${current_explanation.explanation_name}</th>`;
			}


		explanation_result_table_html = explanation_result_table_html +`</tr>
							  </thead>
							  <tbody id="explanation-table-body__${dataset_identifier}__${image_identifier}" class="explanation-table-body">`;	

		//create rows							  
		for (var model_i = 0; model_i < selected_models.length; model_i++){
			current_model = selected_models[model_i];
			
			explanation_result_table_html = explanation_result_table_html +`<tr>
							      <th class="explanation_row_header" id="explanation_row_header__${dataset_identifier}__${image_identifier}__${current_model.class_name}" scope="row">${current_model.model_name}</th>`;
							      
							     
			//create cells
			for (var explanation_i = 0; explanation_i < selected_explanations.length; explanation_i++){
				current_explanation = selected_explanations[explanation_i];
				 	explanation_result_table_html = explanation_result_table_html +`<td class="explanation_result" id="explanation_result__${dataset_identifier}__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}" >
						      	<img id="explanation-table-result-image__${dataset_identifier}__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}" class="explanation-table-result-image" src="test_content/testset_1_preview.jpg">
						      	<span class="explanation-table-result-image-prediction" id="explanation-table-result-image-prediction__${dataset_identifier}__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}">Fetching...</span>
							    <span class="explanation-table-result-text" id="explanation-table-result-text__${dataset_identifier}__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}"></span>
							    <span class="explanation-table-result-json-storage json-storage" id="explanation-table-result-json-storage__${dataset_identifier}__${image_identifier}__${current_model.class_name}__${current_explanation.class_name}">{}</span>
							    
							     
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
}

function PopulateExplanationTable(dataset_name,input_image,model,explanation)
{
	GetExplanationForTable(dataset_name,model,explanation,input_image,false,AddExplanationResultToTable)
}

function AddExplanationResultToTable(explanation_result,dataset_name,image_identifier,model_identifier,explanation_identifier)
{	
	var dataset_identifier = dataset_name.split(" ").join("_").replace("(","").replace(")","");

	var image_element_id = `explanation-table-result-image__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+image_element_id).attr("src","data:image/jpg;base64,"+explanation_result["explanation_image"]);

	var ground_truth_id = `explanation-table-result-image-prediction__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+ground_truth_id).html(explanation_result["prediction"]);

	var ground_truth_id = `explanation-table-result-text__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+ground_truth_id).html(explanation_result["explanation_text"]);

	

	var json_id = `explanation-table-result-json-storage__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+json_id).html(JSON.stringify(explanation_result));

}


function GetExplanationForTable(dataset_name,model,explanation,input_image,attribution_map,callback_function)
{
	var image_name = input_image.image_name;
	var image_identifier = input_image.image_name.replace(".jpg","").replace(".JPEG","");
	var model_identifier = model.class_name;
	var explanation_identifier = explanation.class_name;

	var explain_url = api_base_url + `/explanation-explain?dataset=${dataset_name}&model=${model.model_name}&image=${image_name}&explanation=${explanation.explanation_name}&attribution_map=${attribution_map}`;

	$.ajax({
	        url: explain_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	callback_function(data,dataset_name,image_identifier,model_identifier,explanation_identifier);          
	        }
	    })
}




function PopulateExplanationTableSynchronously(dataset_name,input_images,selected_models,selected_explanations)
{
	GetExplanationForTableAndFetchNext(dataset_name,false,AddExplanationResultToTableAndFetchNext,input_images,selected_models,selected_explanations,0,0,0);
}

function AddExplanationResultToTableAndFetchNext(explanation_result,image_identifier,model_identifier,explanation_identifier,dataset_name,input_images,selected_models,selected_explanations,image_i,model_i,explanation_i)
{	
	var dataset_identifier = dataset_name.split(" ").join("_").replace("(","").replace(")","");

	var image_element_id = `explanation-table-result-image__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+image_element_id).attr("src","data:image/jpg;base64,"+explanation_result["explanation_image"]);

	var ground_truth_id = `explanation-table-result-image-prediction__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+ground_truth_id).html(explanation_result["prediction"]);

	var ground_truth_id = `explanation-table-result-text__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	$("#"+ground_truth_id).html(explanation_result["explanation_text"]);


	var json_id = `explanation-table-result-json-storage__${dataset_identifier}__${image_identifier}__${model_identifier}__${explanation_identifier}`;
	// $("#"+json_id).html(JSON.stringify({"dataset_identifier":dataset_identifier,"image_identifier":image_identifier,"model_identifier":model_identifier,"explanation_identifier":explanation_identifier,"explanation_result":explanation_result}));
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

	var image_identifier = input_image.image_name.replace(".jpg","").replace(".JPEG","");
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




function OLDProduceJsonForStorage()
{
	$("#results_string_output").html("Generating JSON for download...");
	
	var explanation_results = $(".explanation_result");

	var json_strings = [];

	for (var result_i = 0; result_i < explanation_results.length; result_i++){
		var current_result = explanation_results[result_i];
		var result_id = current_result.id.replace("explanation_result__","");
		var result_identifiers = result_id.split("__");

		var result_dict = {};

		result_dict["dataset_identifier"] = result_identifiers[0]
		result_dict["image_identifier"] = result_identifiers[1];
		result_dict["model_identifier"] = result_identifiers[2];
		result_dict["explanation_identifier"] = result_identifiers[3];
		result_dict["result_json"] = JSON.parse($("#explanation-table-result-json-storage__"+result_id).html());

		var json_string = JSON.stringify(result_dict);

		json_strings.push(json_string);
	}
	//$("#results_string_output").html('{"results":['+json_strings.join(",<br>")+']}');
	
	

	//DownloadTableJson('{"explanation_table_scaffold":'+$("#explanation_table_json_storage").html()+',"explanation_table_data":{"results":[\n'+json_strings.join(",\n")+']} }');
	
}

function ProduceJsonForStorage()
{
	var experiment_id = "experiment_" + Date.now().toString()

	var create_folder_url = base_data_api_url + "create_table_folder/"+experiment_id	

	$.ajax({
	        url: create_folder_url,
	        type: "get",
	        success: function (data) 
	        {
	        	var explanation_results = $(".explanation_result");

				SendResultToSaveAPI(explanation_results,0,data);

			}
	});

}


function SendResultToSaveAPI(explanation_results,result_i,experiment_path)
{
	var save_json_url = base_data_api_url + "save_explanation_result_json"


	var current_result = explanation_results[result_i];
	var result_id = current_result.id.replace("explanation_result__","");
	var result_identifiers = result_id.split("__");

	var result_dict = {};

	result_dict["dataset_identifier"] = result_identifiers[0]
	result_dict["image_identifier"] = result_identifiers[1];
	result_dict["model_identifier"] = result_identifiers[2];
	result_dict["explanation_identifier"] = result_identifiers[3];
	result_dict["result_json"] = JSON.parse($("#explanation-table-result-json-storage__"+result_id).html());

	var json_string = JSON.stringify(result_dict);

	var data_string = JSON.stringify({"experiment_path":experiment_path,"explanation":json_string});
	$.ajax({
        url: save_json_url,
        type: "POST",
        crossDomain: true,
        data:data_string,
        success: function (data) 
        {
        	result_i++;
        	if(result_i < explanation_results.length)
        	{
        		SendResultToSaveAPI(explanation_results,result_i,experiment_path);
        	}
        	else
        	{
        		CompileJsonFile(experiment_path);
        	}
       	}
    });
	
}

function CompileJsonFile(experiment_path)
{
	var compile_json_url = base_data_api_url + "compile_explanation_table_json"

	$.ajax({
        url: compile_json_url,
        type: "post",
        data:JSON.stringify({"experiment_path":experiment_path,"explanation_table_scaffold":$("#explanation_table_json_storage").html()}),
        success: function (data) 
        {
        	alert("Table saved in:"+data);
        }
    });

}


function SaveTableViaService()
{
	var save_table_url = base_data_api_url+"save_explanation_table_from_json";

	var table_json = '{"explanation_table_scaffold":'+$("#explanation_table_json_storage").html()+',"explanation_table_data":{"results":[\n'+json_strings.join(",\n")+']} }';
	$.ajax({
	        url: save_table_url,
	        type: "post",
	        data:table_json,
	        success: function (data) 
	        {
	        	alert("table html sent to save function");
	       	}
	    });
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

function LoadFromExperimentId()
{
	$("#load_table_from_id_progress_container").show();
	experiment_id = $("#table_json_id_input").val();
	
	var table_scaffold_url = base_data_api_url+"get_table_scaffold/"+experiment_id;

	$.ajax({
	        url: table_scaffold_url,
	        type: "get",
	        crossDomain: true,
            success: function (data) 
	        {
	        	
	        	var scaffold_json = JSON.parse(data);
	        	var table_scaffold = scaffold_json.explanation_table_scaffold;

				CreateTable(table_scaffold.input_images,table_scaffold.dataset_name,table_scaffold.selected_models,table_scaffold.selected_explanations);

				$("#scaffold_done_message").show();
				LoadFetchExplanationsByExperimentId(experiment_id);
	       	}
	    });

}


function LoadFetchExplanationsByExperimentId(experiment_id)
{

	var num_explanations_url = base_data_api_url+"get_number_explanations_for_table/"+experiment_id;

	$("#progress_container").show();
	
	$.ajax({
	        url: num_explanations_url,
	        type: "get",
	        crossDomain: true,
            success: function (data) 
	        {
	        	var num_explanations = parseInt(data);
	        	FetchExplanationByExperimentIdAndIndex(experiment_id,0,num_explanations);
	        	
	       	}
	    });
}


function FetchExplanationByExperimentIdAndIndex(experiment_id,explanation_i,total_explanations)
{
	var fetch_result_url = base_data_api_url + "get_table_explanation_by_index"

	var data_string = JSON.stringify({"experiment_id":experiment_id,"explanation_i":explanation_i});

	$.ajax({
        url: fetch_result_url,
        type: "POST",
        crossDomain: true,
        data:data_string,
        success: function (data) 
        {
        	var result = JSON.parse(data)
        	AddExplanationResultToTable(result.result_json,result.dataset_identifier,result.image_identifier,result.model_identifier,result.explanation_identifier);	
			
			UpdateLoadProgressBar(explanation_i,total_explanations);
        	explanation_i++;
        	if(explanation_i < total_explanations)
        	{
        		FetchExplanationByExperimentIdAndIndex(experiment_id,explanation_i,total_explanations);
        	}
        	else
        	{
        		alert("table loaded");
        	}
       	}
    });
	
}

function UpdateLoadProgressBar(completed_explanation_i,total_explanations)
{
	var complete_percentage = Math.round(((completed_explanation_i+1)/total_explanations)*100);

	$("#load_progress_bar").css('width', complete_percentage+'%').attr('aria-valuenow', complete_percentage);
}


function BuildTableFromJsonString(table_json_string)
{
	var table_json = JSON.parse(table_json_string);

	// $("#card-deck__explanation-table").html(table_json.explanation_table_data.table_html.html);

	var table_scaffold = table_json.explanation_table_scaffold;

	CreateTable(table_scaffold.input_images,table_scaffold.dataset_name,table_scaffold.selected_models,table_scaffold.selected_explanations);

	var results = table_json.explanation_table_data.results;

	for (var result_i = 0; result_i < results.length; result_i++){
		var result = results[result_i];

		AddExplanationResultToTable(result.result_json,result.dataset_identifier,result.image_identifier,result.model_identifier,result.explanation_identifier);	
	}
	
	// LoadItemJsonFromId();
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