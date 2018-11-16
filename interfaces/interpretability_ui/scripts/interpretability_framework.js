
//var api_base_url = "http://services.futurelabsolutions.com:6112";
var api_base_url = "http:localhost:3100";

// DATASETS

function GetDatasets()
{
	all_datasets_url = api_base_url + "/datasets-all";  
	$.ajax({
	        url: all_datasets_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	UpdateAvailableDatasets(data);          
	        }
	    })
}

function UpdateAvailableDatasets(datasets_json)
{
	
	for (var i = 0; i < datasets_json.length; i++){
		var dataset_json = datasets_json[i];
		$('#card-deck-datasets').append(DatasetCardFromJson(dataset_json));

		if(i != 0 && (i+1) % 2 === 0)
		{
			AddCardBreakToElement('#card-deck-datasets');
		}  
	}
	
}


function DatasetCardFromJson(dataset_json)
{
	var dataset_name = dataset_json.dataset_name;
	var dataset_thumbnail_img_name = dataset_json.interesting_images[0];
	
	var dataset_description = dataset_json["description"];
	var dataset_id = dataset_json["folder"];
	
	var dataset_img_src = FectchDatasetImageByName("dataset-item-image__"+dataset_id,"src",dataset_name, dataset_thumbnail_img_name);

	var dataset_card_html = `<div class="card mb-4 col-6" id="dataset-card__${dataset_id}">
	  <img class="dataset-item-image card-img-top" id="dataset-item-image__${dataset_id}" src="${dataset_img_src}">
	  <div class="card-body">
	    <h5 class="card-title">${dataset_name}</h5>
	    <p class="dataset-card-text card-text">${dataset_description}</p>
	    <button type="button" id="dataset-select-button__${dataset_id}" class="btn btn-secondary dataset-select-button" onClick="SelectDataset(this.id)">Select Dataset</button>
	    <div class="json-storage" id="json-storage__${dataset_id}">${JSON.stringify(dataset_json)}</div>
	  </div>
	</div>`;

	return dataset_card_html;
}

function AddCardBreakToElement(element_id)
{
	var break_html = `<div class="w-100 d-block d-md-block d-lg-block"></div>`;
	$(element_id).append(break_html); 
}


function SelectDataset(dataset_button_id)
{
	var dataset_id = dataset_button_id.split("__")[1];
	var dataset_json = LoadItemJsonFromId(dataset_id);

	$("#select-item-id-storage__image").html(dataset_json.interesting_images[0]);
	$("#selected-image-name").html(dataset_json.interesting_images[0]);

	//deselect the current selected dataset
	var previous_dataset_id = $("#select-item-id-storage__datasets").html();
	if(previous_dataset_id !== "")
	{
		DeselectDataset(previous_dataset_id);
	}

	//update datasets title display to show selected dataset
	//update datasets title display class to change colour
	$( "#selection-name-display__datasets" ).removeClass( "badge-danger" ).addClass( "badge-success" ).html(dataset_json.dataset_name);
	
	// update card border colour
	$( "#dataset-card__"+dataset_id ).addClass("selected-card");
	
	//update button text to show "deselect"
	//update button class to change colour
	//update button onClick to deselect if clicked again
	$( "#dataset-select-button__"+dataset_id ).removeClass( "btn-secondary" ).addClass( "btn-success" ).html("Deselect Dataset");
	
	$( "#dataset-select-button__"+dataset_id ).prop("onclick", null).off("click");
	$( "#dataset-select-button__"+dataset_id ).on('click',function(){DeselectDataset(dataset_id);});
	
	$("#select-item-id-storage__datasets").html(dataset_id);

	//close fold and open models fold
	// $("#section-expansion-symbol__models").trigger("click");
}

function DeselectDataset(dataset_id)
{
	//update datasets title display to show selected dataset
	//update datasets title display class to change colour
	$( "#selection-name-display__datasets" ).removeClass( "badge-success" ).addClass( "badge-danger" ).html("None Selected");
	
	// update card border colour
	$( "#dataset-card__"+dataset_id ).removeClass("selected-card");
	
	//update button text to show "deselect"
	//update button class to change colour
	//update button onClick to deselect if clicked again
	$( "#dataset-select-button__"+dataset_id ).addClass( "btn-secondary" ).removeClass( "btn-success" ).html("Select Dataset");
	
	$( "#dataset-select-button__"+dataset_id ).prop("onclick", null).off("click");
	$( "#dataset-select-button__"+dataset_id ).on('click',function(){SelectDataset("dataset-select-button__"+dataset_id);});
	
	$("#select-item-id-storage__datasets").html("");
}






function FectchDatasetImageByName(element_id,destination_property,dataset_name, image_name)
{	
	var fetch_image_url = api_base_url + `/dataset-test-image?dataset=${dataset_name}&image=${image_name}`;
	$.ajax({
	        url: fetch_image_url,
	        type: "GET",
	        success: function (data) 
	        {	
	        	$("#"+element_id).attr(destination_property, "data:image/jpg;base64,"+data.input);       
	        }
	    })
	return "test_content/testset_1_preview.jpg";

}

// MODELS 
function GetModels()
{
	all_datasets_url = api_base_url + "/models-all";  
	$.ajax({
	        url: all_datasets_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	UpdateAvailableModels(data);          
	        }
	    })
}

function UpdateAvailableModels(json)
{
	
	for (var i = 0; i < json.length; i++){
		var model_json = json[i];
		if(CheckModelIsTrainedOnDataset(model_json))
		{
			$('#models_lists').append(ModelListItemFromJson(model_json));
  		}
		
	}
	
}

function CheckModelIsTrainedOnDataset(model_json)
{
	return true;
}


function ModelListItemFromJson(model_json)
{
	var model_name = model_json.model_name;
	var model_id = model_json["class_name"];
	
	var model_list_item_html = `<li class="list-group-item" id="model-list-item__${model_id}"> 
		<div class="row">
			<div class="model-item-name-container col" id="model-item-name-container__${model_id}">
				<span class="model-item-name" id="model-item-name__test_model_1">${model_name}</span>
			</div> 
			<div class="model-item-toggle-container col-1" id="model-item-toggle-container__${model_id}">
				<button type="button" class="model-toggle-button btn btn-secondary" id="model-toggle-button__${model_id}" onClick="SelectModel(this.id)">Select Model</button>
			</div>
			<div class="json-storage" id="json-storage__${model_id}">${JSON.stringify(model_json)}</div>
	  </div> 
	</li>`;

	return model_list_item_html;
}



function SelectModel(model_button_id)
{
	var model_id = model_button_id.split("__")[1];
	var model_json = LoadItemJsonFromId(model_id);

	//deselect the current selected model
	// var previous_model_id = $("#select-item-id-storage__models").html();
	// if(previous_model_id !== "")
	// {
	// 	DeselectModel(previous_model_id);
	// }

	//update models title display to show selected model
	//update models title display class to change colour
	var new_selection_display = $( "#selection-name-display__models" ).html();
	if(new_selection_display === "None Selected")
	{
		new_selection_display = "";	
	}
	new_selection_display = new_selection_display+model_id +" | ";
	
	$( "#selection-name-display__models" ).removeClass( "badge-danger" ).addClass( "badge-success" ).html(new_selection_display);
	
	//update button text to show "deselect"
	//update button class to change colour
	//update button onClick to deselect if clicked again
	$( "#model-toggle-button__"+model_id ).removeClass( "btn-secondary" ).addClass( "btn-success" ).html("Deselect model");
	
	$( "#model-toggle-button__"+model_id ).prop("onclick", null).off("click");
	$( "#model-toggle-button__"+model_id ).on('click',function(){DeselectModel(model_id);});
	
	// $("#select-item-id-storage__models").html(model_id);
	var new_selection_storage = $("#select-item-id-storage__models").html();
	new_selection_storage = new_selection_storage+model_id +"|";
	
	$("#select-item-id-storage__models").html(new_selection_storage);

	//close fold and open models fold
	// $("#section-expansion-symbol__models").trigger("click");
}


function DeselectModel(model_id)
{
	//update models title display to show selected model
	//update models title display class to change colour
	var new_selection_display = $( "#selection-name-display__models" ).html();
	new_selection_display = new_selection_display.replace(model_id +" | ","");
	if(new_selection_display === "")
	{
		$( "#selection-name-display__models" ).removeClass( "badge-success" ).addClass( "badge-danger" ).html("None Selected");
	}
	else
	{
		$( "#selection-name-display__models" ).html(new_selection_display);
	}

	//update button text to show "deselect"
	//update button class to change colour
	//update button onClick to deselect if clicked again
	$( "#model-toggle-button__"+model_id ).addClass( "btn-secondary" ).removeClass( "btn-success" ).html("Select model");
	
	$( "#model-toggle-button__"+model_id ).prop("onclick", null).off("click");
	$( "#model-toggle-button__"+model_id ).on('click',function(){SelectModel("model-select-button__"+model_id);});
	
	var new_selection_storage = $("#select-item-id-storage__models").html();
	new_selection_storage = new_selection_storage.replace(model_id +"|","");
	
	$("#select-item-id-storage__models").html(new_selection_storage);
}




// MODELS 
function GetExplanations()
{
	all_datasets_url = api_base_url + "/explanations-all";  
	$.ajax({
	        url: all_datasets_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	UpdateAvailableExplanations(data);          
	        }
	    })
}

function UpdateAvailableExplanations(json)
{
	
	for (var i = 0; i < json.length; i++){
		var explanation_json = json[i];
		if(CheckExplanationIsTrainedOnDataset(explanation_json))
		{
			$('#explanations_lists').append(ExplanationListItemFromJson(explanation_json));
  		}
		
	}
	
}

function CheckExplanationIsTrainedOnDataset(explanation_json)
{
	return true;
}


function ExplanationListItemFromJson(explanation_json)
{
	var explanation_name = explanation_json.explanation_name;
	var explanation_id = explanation_json["class_name"];
	
	var explanation_list_item_html = `<li class="list-group-item" id="explanation-list-item__${explanation_id}"> 
		<div class="row">
			<div class="explanation-item-name-container col" id="explanation-item-name-container__${explanation_id}">
				<span class="explanation-item-name" id="explanation-item-name__test_explanation_1">${explanation_name}</span>
			</div> 
			<div class="explanation-item-toggle-container col-1" id="explanation-item-toggle-container__${explanation_id}">
				<button type="button" class="explanation-toggle-button btn btn-secondary" id="explanation-toggle-button__${explanation_id}" onClick="SelectExplanation(this.id)">Select Explanation</button>
			</div>
			<div class="json-storage" id="json-storage__${explanation_id}">${JSON.stringify(explanation_json)}</div>
	  </div> 
	</li>`;

	return explanation_list_item_html;
}



function SelectExplanation(explanation_button_id)
{
	var explanation_id = explanation_button_id.split("__")[1];
	var explanation_json = LoadItemJsonFromId(explanation_id);

	//deselect the current selected explanation
	// var previous_explanation_id = $("#select-item-id-storage__explanations").html();
	// if(previous_explanation_id !== "")
	// {
	// 	DeselectExplanation(previous_explanation_id);
	// }

	//update explanations title display to show selected explanation
	//update explanations title display class to change colour
	var new_selection_display = $( "#selection-name-display__explanations" ).html();
	if(new_selection_display === "None Selected")
	{
		new_selection_display = "";	
	}
	new_selection_display = new_selection_display+explanation_id +" | ";
	
	$( "#selection-name-display__explanations" ).removeClass( "badge-danger" ).addClass( "badge-success" ).html(new_selection_display);
	
	//update button text to show "deselect"
	//update button class to change colour
	//update button onClick to deselect if clicked again
	$( "#explanation-toggle-button__"+explanation_id ).removeClass( "btn-secondary" ).addClass( "btn-success" ).html("Deselect explanation");
	
	$( "#explanation-toggle-button__"+explanation_id ).prop("onclick", null).off("click");
	$( "#explanation-toggle-button__"+explanation_id ).on('click',function(){DeselectExplanation(explanation_id);});
	
	// $("#select-item-id-storage__explanations").html(explanation_id);
	var new_selection_storage = $("#select-item-id-storage__explanations").html();
	new_selection_storage = new_selection_storage+explanation_id +"|";
	
	$("#select-item-id-storage__explanations").html(new_selection_storage);

	//close fold and open explanations fold
	// $("#section-expansion-symbol__explanations").trigger("click");
}


function DeselectExplanation(explanation_id)
{
	//update explanations title display to show selected explanation
	//update explanations title display class to change colour
	var new_selection_display = $( "#selection-name-display__explanations" ).html();
	new_selection_display = new_selection_display.replace(explanation_id +" | ","");
	if(new_selection_display === "")
	{
		$( "#selection-name-display__explanations" ).removeClass( "badge-success" ).addClass( "badge-danger" ).html("None Selected");
	}
	else
	{
		$( "#selection-name-display__explanations" ).html(new_selection_display);
	}

	//update button text to show "deselect"
	//update button class to change colour
	//update button onClick to deselect if clicked again
	$( "#explanation-toggle-button__"+explanation_id ).addClass( "btn-secondary" ).removeClass( "btn-success" ).html("Select explanation");
	
	$( "#explanation-toggle-button__"+explanation_id ).prop("onclick", null).off("click");
	$( "#explanation-toggle-button__"+explanation_id ).on('click',function(){SelectExplanation("explanation-select-button__"+explanation_id);});
	
	var new_selection_storage = $("#select-item-id-storage__explanations").html();
	new_selection_storage = new_selection_storage.replace(explanation_id +"|","");
	
	$("#select-item-id-storage__explanations").html(new_selection_storage);
}



// MAKING EXPLAIN CALL

function GetActiveDatasetName()
{
	var dataset_id = $("#select-item-id-storage__datasets").html();
	var dataset_json = LoadItemJsonFromId(dataset_id);
	return dataset_json["dataset_name"];
}


function GetActiveModelNames()
{

	var selected_model_ids = $("#select-item-id-storage__models").html().split("|");
	
	var model_names = [];
	for (var i = 0; i < selected_model_ids.length; i++){
		if(selected_model_ids[i] != "")
		{
			var model_json = LoadItemJsonFromId(selected_model_ids[i]);
			model_names.push(model_json["model_name"]);
		}
		
	}
	return model_names
}


function GetActiveModels()
{

	var selected_model_ids = $("#select-item-id-storage__models").html().split("|");
	
	var selected_models = [];
	for (var i = 0; i < selected_model_ids.length; i++){
		if(selected_model_ids[i] != "")
		{
			var model_json = LoadItemJsonFromId(selected_model_ids[i]);
			selected_models.push(model_json);
		}
		
	}
	return selected_models
}


function GetActiveExplanations()
{

	var selected_explanation_ids = $("#select-item-id-storage__explanations").html().split("|");
	
	var selected_explanations = [];
	for (var i = 0; i < selected_explanation_ids.length; i++){
		if(selected_explanation_ids[i] != "")
		{
			var explanation_json = LoadItemJsonFromId(selected_explanation_ids[i]);
			selected_explanations.push(explanation_json);
		}
	}
	return selected_explanations
}

function GetActiveExplanationNames()
{

	var selected_explanation_ids = $("#select-item-id-storage__explanations").html().split("|");
	
	var explanation_names = [];
	for (var i = 0; i < selected_explanation_ids.length; i++){
		if(selected_explanation_ids[i] != "")
		{
			var explanation_json = LoadItemJsonFromId(selected_explanation_ids[i]);
			explanation_names.push(explanation_json["explanation_name"]);
		}
	}
	return explanation_names
}


function Explain()
{
	var dataset_name = GetActiveDatasetName();
	
	var selected_model_names = GetActiveModelNames();
	var model_name = selected_model_names[0];
	
	var selected_explanation_names = GetActiveExplanationNames();
	var explanation_name = selected_explanation_names[0];
	
	var image_name = $("#select-item-id-storage__image").html();
	
	var attribution_map = false;

	GetExplanation(dataset_name,model_name,explanation_name,image_name,attribution_map,AddExplanationToResults)
}

function GetExplanation(dataset_name,model_name,explanation_name,image_name,attribution_map,callback_function)
{
	var explain_url = api_base_url + `/explanation-explain?dataset=${dataset_name}&model=${model_name}&image=${image_name}&explanation=${explanation_name}&attribution_map=${attribution_map}`;
	// explain_url = explain_url.split(" ").join("%20");
	$.ajax({
	        url: explain_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	callback_function(data);          
	        }
	    })
}


function AddExplanationToResults(result_json)
{
	var result_html = `<div class="card col-3 mb-4">
								  <img class="result-image card-img-top" src="data:image/jpg;base64, ${result_json["explanation_image"]}">
								</div>`
	$('#result-card-deck').append(result_html);

	var total_results = parseInt($('#var_storage__num_results').html());
	total_results++;
	$('#var_storage__num_results').html(total_results);

	if(total_results % 4 == 0)
	{
		AddCardBreakToElement('#result-card-deck');
	}
}



function LoadItemJsonFromId(item_id)
{
	var json_string = $("#json-storage__"+item_id).html();
	return JSON.parse(json_string);
}


function ChangeDatasetBrowseTab(browse_link_id)
{
	var class_id = browse_link_id.split("__")[1];
	// (dataset-browse-link__class_1)

	var tab_id = "dataset-browse-card-body__"+class_id;

	var current_tab_id = $("#dataset-browse-active-tab").html();

	$("#"+current_tab_id).addClass("dataset-browse-card-body-inactive");
	$("#"+tab_id).removeClass("dataset-browse-card-body-inactive");
	$("#dataset-browse-active-tab").html(tab_id);
	// dataset-browse-card-body__class_1
}

function DatasetBrowseSelectImage(dataset_card_id)
{
	var img_id = dataset_card_id.split("__")[1];
	var image_name_storage_id = "dataset-browse-img-card-name__"+img_id;

	var image_name = $("#"+image_name_storage_id).html();

	var image_elemet_id = "dataset-browse-img__"+img_id;

	var image_src = $("#"+image_elemet_id).attr("src");

	var preview_img_id = "input-image"
	$("#"+preview_img_id).attr("src",image_src);

	var browse_close_button_id = "close-browse-dataset-button";
	$("#"+browse_close_button_id).click();

	var current_image_name_storage_id = "select-item-id-storage__image";
	$("#"+current_image_name_storage_id).html(image_name);
}


function GetDatasetBrowseImages(dataset_name)
{

	browse_dataset_url = api_base_url + `dataset-test-images?dataset=${dataset_name}&num_images=100`;  
	$.ajax({
	        url: browse_dataset_url,
	        type: "GET",
	        success: function (data) 
	        {
	        	UpdateBrowseDatasetsPanel(data);          
	        }
	    })
}


function UpdateBrowseDatasetsPanel(dataset_images)
{
	var browse_image_dict = {};
	browse_image_dict["all_classes"] = [];

	for (var i = 0; i < dataset_images.length; i++){
		browse_image = dataset_images[i];

		if(! browse_image.ground_truth in browse_image_dict){
			browse_image_dict[browse_image.ground_truth] = [];
		}

		browse_image_dict["all_classes"].push(browse_image);
		browse_image_dict[browse_image.ground_truth].push(browse_image);
	}

	var browse_link_list_id = "dataset-browse-links-list";
	$("#"+browse_link_list_id).html("");

}


function CreateBrowseDatasetCard(image_name,img_src)
{
	var image_id = image_name.replace(".jpg","");

	var card_html = `<div class="card mb-3 col-2 d-flex align-items-stretch dataset-browse-img-card" id="dataset-browse-img-card__${image_id}" onclick="DatasetBrowseSelectImage(this.id)">
		<div class="var_storage" id="dataset-browse-img-card-name__${image_id}">${image_name}</div>
		<img class="result-image card-img-top" id="dataset-browse-img__${image_id}" src="data:image/jpg;base64,${img_src}">
	</div>`;

	return card_html


}


function PopulateChoices(dataset_identifier,model_identifiers,explanation_identifiers)
{
	$("#dataset-select-button__"+dataset_identifier).click();

	for (var i = 0; i < model_identifiers.length; i++){
		current_model_identifier = model_identifiers[i];

		$("#model-toggle-button__"+current_model_identifier).click();
	}


	for (var i = 0; i < explanation_identifiers.length; i++){
		current_explanation_identifier = explanation_identifiers[i];

		$("#explanation-toggle-button__"+current_explanation_identifier).click();
	}



	
}

// QUICK POPULATE FUNCTIONS


function PopulateGUNVGGALL()
{
	PopulateChoices("wielder_non-wielder",["VGG16Imagenet"],["LimeExplainer","ShapExplainer","LRPExplainer","InfluenceExplainer"]);
}

function PopulateTFLrVGGALL()
{
	PopulateChoices("congested_non-congested_resized",["VGG16Imagenet"],["LimeExplainer","ShapExplainer","LRPExplainer","InfluenceExplainer"]);
}

function PopulateTFLrVGGLIME()
{
	PopulateChoices("congested_non-congested_resized",["VGG16Imagenet"],["LimeExplainer"]);
}

