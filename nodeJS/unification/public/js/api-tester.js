// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let globals = {
    "datasets": [],
    "models": [],
    "explanations": [],
    "images": [],
    "images_no_default": [],
    "number_of_images": 10,
    "mappings": []
};

function showAPIs() {
    populateLists();
    uiConfig();
    renderLists();
    recalculateUrls();
    showSections();
}

function showSections() {
    let e = document.getElementById("div_datasets");
    e.style.display = "block";

    e = document.getElementById("div_models");
    e.style.display = "block";

    e = document.getElementById("div_explanations");
    e.style.display = "block";
}

function populateLists() {
    populateTypes();
    populateImages();
    listDatasets();
    listModels();
    listExplanations();

    console.log(globals);
}

function uiConfig() {
    globals.uiConfig = {};
    globals.uiConfig.ds_all = { "rootUi": "ds_all", "rootUrl": "datasets-all", "uiLists": [] };
    globals.uiConfig.ds_details = { "rootUi": "ds_details", "rootUrl": "dataset-details", "uiLists": [] };
    globals.uiConfig.ds_img = { "rootUi": "ds_img", "rootUrl": "dataset-test-image", "uiLists": [] };
    globals.uiConfig.ds_imgs = { "rootUi": "ds_imgs", "rootUrl": "dataset-test-images", "uiLists": [] };
    globals.uiConfig.ds_arch = { "rootUi": "ds_arch", "rootUrl": "dataset-archive", "uiLists": [] };
    globals.uiConfig.mod_all = { "rootUi": "mod_all", "rootUrl": "models-all", "uiLists": [] };
    globals.uiConfig.mod_details = { "rootUi": "mod_details", "rootUrl": "model-details", "uiLists": [] };
    globals.uiConfig.mod_fords = { "rootUi": "mod_fords", "rootUrl": "models-for-dataset", "uiLists": [] };
    globals.uiConfig.mod_pred = { "rootUi": "mod_pred", "rootUrl": "model-predict", "uiLists": [] };
    globals.uiConfig.mod_arch = { "rootUi": "mod_arch", "rootUrl": "model-archive", "uiLists": [] };
    globals.uiConfig.exp_all = { "rootUi": "exp_all", "rootUrl": "explanations-all", "uiLists": [] };
    globals.uiConfig.exp_details = { "rootUi": "exp_details", "rootUrl": "explanation-details", "uiLists": [] };
    globals.uiConfig.exp_filt = { "rootUi": "exp_filt", "rootUrl": "explanations-for-filter", "uiLists": [] };
    globals.uiConfig.exp_exp = { "rootUi": "exp_exp", "rootUrl": "explanation-explain", "uiLists": [] };
    globals.uiConfig.exp_atm = { "rootUi": "exp_atm", "rootUrl": "explanation-attribution-map", "uiLists": [] };

    globals.mappings.push( [ "ds_img", "ds_list", "img_list", "interesting_images", globals.images ] );
    globals.mappings.push( [ "mod_pred", "ds_list", "img_list", "interesting_images", globals.images_no_default ] );
    globals.mappings.push( [ "exp_exp", "ds_list", "img_list", "interesting_images", globals.images_no_default ] );
    globals.mappings.push( [ "exp_atm", "ds_list", "img_list", "interesting_images", globals.images_no_default ] );

    let uiTypes = [ "type_list", "type", null, globals.types ];
    let uiNumImages = [ "num_images", "num_images", null, globals.number_of_images ];
    let uiDatasets = [ "ds_list", "dataset", "dataset_name", globals.datasets ];
    let uiModels = [ "mod_list", "model", "model_name", globals.models ];
    let uiExplanations = [ "exp_list", "explanation", "explanation_name", globals.explanations ];
    let uiImages = [ "img_list", "image", null, globals.images ];
    let uiImagesNoDefault = [ "img_list", "image", null, globals.images_no_default ];

    globals.uiConfig.ds_all.uiLists.push(uiTypes);

    globals.uiConfig.ds_details.uiLists.push(uiTypes);
    globals.uiConfig.ds_details.uiLists.push(uiDatasets);

    globals.uiConfig.ds_img.uiLists.push(uiTypes);
    globals.uiConfig.ds_img.uiLists.push(uiDatasets);
    globals.uiConfig.ds_img.uiLists.push(uiImages);

    globals.uiConfig.ds_imgs.uiLists.push(uiTypes);
    globals.uiConfig.ds_imgs.uiLists.push(uiDatasets);
    globals.uiConfig.ds_imgs.uiLists.push(uiNumImages);

    globals.uiConfig.ds_arch.uiLists.push(uiDatasets);

    globals.uiConfig.mod_all.uiLists.push(uiTypes);

    globals.uiConfig.mod_details.uiLists.push(uiTypes);
    globals.uiConfig.mod_details.uiLists.push(uiModels);

    globals.uiConfig.mod_fords.uiLists.push(uiTypes);
    globals.uiConfig.mod_fords.uiLists.push(uiDatasets);

    globals.uiConfig.mod_pred.uiLists.push(uiTypes);
    globals.uiConfig.mod_pred.uiLists.push(uiDatasets);
    globals.uiConfig.mod_pred.uiLists.push(uiModels);
    globals.uiConfig.mod_pred.uiLists.push(uiImagesNoDefault);

    globals.uiConfig.mod_arch.uiLists.push(uiDatasets);
    globals.uiConfig.mod_arch.uiLists.push(uiModels);

    globals.uiConfig.exp_all.uiLists.push(uiTypes);

    globals.uiConfig.exp_details.uiLists.push(uiTypes);
    globals.uiConfig.exp_details.uiLists.push(uiExplanations);

    globals.uiConfig.exp_filt.uiLists.push(uiTypes);
    globals.uiConfig.exp_filt.uiLists.push(uiDatasets);
    globals.uiConfig.exp_filt.uiLists.push(uiModels);

    globals.uiConfig.exp_exp.uiLists.push(uiTypes);
    globals.uiConfig.exp_exp.uiLists.push(uiDatasets);
    globals.uiConfig.exp_exp.uiLists.push(uiModels);
    globals.uiConfig.exp_exp.uiLists.push(uiImagesNoDefault);
    globals.uiConfig.exp_exp.uiLists.push(uiExplanations);

    globals.uiConfig.exp_atm.uiLists.push(uiTypes);
    globals.uiConfig.exp_atm.uiLists.push(uiDatasets);
    globals.uiConfig.exp_atm.uiLists.push(uiModels);
    globals.uiConfig.exp_atm.uiLists.push(uiImagesNoDefault);
    globals.uiConfig.exp_atm.uiLists.push(uiExplanations);
}

function renderLists() {
    for (let i in globals.uiConfig) {
        let thisConf = globals.uiConfig[i];

        for (let j in thisConf.uiLists) {
            let thisUiList = thisConf.uiLists[j];
            let elemName = thisConf.rootUi + "-" + thisUiList[0];
            let keyName = thisUiList[2];
            let e = document.getElementById(elemName);

            if (e != null) {
                if (e.options != null) {
                    clearAllOptions(e);

                    for (let k in thisUiList[3]) {
                        let option = document.createElement("option");
                        let thisVal = null

                        if (keyName == null) {
                            thisVal = thisUiList[3][k];
                        } else {
                            thisVal = [ thisUiList[3][k][keyName], thisUiList[3][k][keyName] ];
                        }

                        option.value = thisVal[0];
                        option.text = thisVal[1];
                        e.add(option);
                    }
                } else {
                    e.value = thisUiList[3];
                }
            } else {
                console.log("Unable to find element " + elemName);
            }
        }
    }
}

function clearAllOptions(e) {
    for(let i = e.options.length - 1; i >= 0; i--)
    {
        e.remove(i);
    }
}

function populateTypes() {
    globals.types = [];
    globals.types.push( ["", "(default - json)"] );
    globals.types.push( ["json", "json"] );
    globals.types.push( ["html", "html"] );
}

function populateImages() {
    globals.images = [];
    globals.images.push( ["", "(default - random image)"] );

    globals.images_no_default = [];
}

function recalculateUrls() {
    for (let i in globals.uiConfig) {
        let thisUiConfig = globals.uiConfig[i];
        recalculate(thisUiConfig.rootUi, thisUiConfig.rootUrl, thisUiConfig.uiLists);
    }

    for (let i in globals.mappings) {
        let thisMapping = globals.mappings[i];
        recalculateMapping(thisMapping);
    }
}

function recalculateMapping(map) {
    let rootElem = map[0];
    let srcPart = map[1];
    let tgtPart = map[2];
    let srcProp = map[3];
    let defaultVals = map[4];

    let srcElem = document.getElementById(rootElem + "-" + srcPart);
    let tgtElem = document.getElementById(rootElem + "-" + tgtPart);
    let selOption = srcElem.options[srcElem.selectedIndex];
    let srcVal = null;
    let srcDs = null;
    let allVals = defaultVals.slice();

    if (selOption != null) {
        srcVal = selOption.value;
        srcDs = getMatchingDataset(srcVal);

        if (srcDs != null) {
            for (let i in srcDs[srcProp]) {
                let thisVal = srcDs[srcProp][i];

                allVals.push([ thisVal, thisVal ]);
            }
        }
    }

    if (tgtElem != null) {
        clearAllOptions(tgtElem);

        for (let i in allVals) {
            let thisVal = allVals[i];
            let option = document.createElement("option");
            option.value = thisVal[0];
            option.text = thisVal[1];
            tgtElem.add(option);
        }
    }
}

function getMatchingDataset(dsName) {
    let result = null;

    for (let i in globals.datasets) {
        let thisDs = globals.datasets[i];

        if (thisDs.dataset_name == dsName) {
            result = thisDs;
        }
    }

    return result;
}

function recalculate(rootElem, rootUrl, parmConf) {
    let eU = document.getElementById(rootElem + "-url");
    let tgtUrl = rootUrl;
    let connector = "?";

    for (let i in parmConf) {
        let elemPart = parmConf[i][0];
        let parmName = parmConf[i][1];
        let elemName = rootElem + "-" + elemPart;

        let e = document.getElementById(elemName);
        let selVal = e.value;

        if (selVal !== "") {
            tgtUrl += connector + parmName + "=" + selVal;
            connector = "&";
        }
    }

    eU.href = tgtUrl;
    eU.innerText = tgtUrl;
}

function listDatasets() {
    let url = "/datasets-all?type=json";
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let dsJson = JSON.parse(xmlHttp.responseText).datasets;
                Array.prototype.push.apply(globals.datasets, dsJson);
                renderLists();
                recalculateUrls();

            } else {
                alert("The list of datasets failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function listModels() {
    let url = "/models-all?type=json";
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let modJson = JSON.parse(xmlHttp.responseText).models;
                Array.prototype.push.apply(globals.models, modJson);
                renderLists();
                recalculateUrls();
            } else {
                alert("The list of models failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function listExplanations() {
    let url = "/explanations-all?type=json";
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let expJson = JSON.parse(xmlHttp.responseText).explanations;
                Array.prototype.push.apply(globals.explanations, expJson);
                renderLists();
                recalculateUrls();
            } else {
                alert("The list of explanations failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}
