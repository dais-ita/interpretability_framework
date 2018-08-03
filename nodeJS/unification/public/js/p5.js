// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

function useSelectedDataset() {
    let url = "/models-for-dataset?type=json&dataset=" + getSelectedDatasetName();

    populateModels(url);
}

function populateExplanations() {
    let url = "";
    let xmlHttp = new XMLHttpRequest();

    url += "/explanations-for-filter?";
    url += "type=json";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsExps = JSON.parse(xmlHttp.responseText);

                let eXl = document.getElementById("exp_list");
                clearAllOptions(eXl);

                for (let i in jsExps.explanations) {
                    let thisExp = jsExps.explanations[i];
                    let option = document.createElement("option");
                    option.text = thisExp.explanation_name;
                    eXl.add(option);
                }
            } else {
                alert("The list of explanation types failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function populateModels(url) {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsMods = JSON.parse(xmlHttp.responseText);

                let eSm = document.getElementById("settings_model");
                let eMl = document.getElementById("mod_list");
                clearAllOptions(eMl);

                eSm.style.display = "block";

                for (let i in jsMods.models) {
                    let thisMod = jsMods.models[i];
                    let option = document.createElement("option");
                    option.text = thisMod.model_name;
                    eMl.add(option);
                }
            } else {
                alert("The list of models failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function clearAllOptions(e) {
    for(let i = e.options.length - 1; i >= 0; i--)
    {
        e.remove(i);
    }
}

function useSelectedModel() {
    let eSi = document.getElementById("settings_image");

    eSi.style.display = "block";

    populateExplanations();
}

function useRandomImage() {
    let url = "/datasets-test-image?type=json&dataset=" + getSelectedDatasetName();

    requestImage(url);
}

function usePredefinedImage(imageName) {
    let url = "/datasets-test-image?type=json&dataset=" + getSelectedDatasetName() + "&image=" + imageName;

    requestImage(url);
}

function getSelectedDatasetName() {
    let e = document.getElementById("ds_list");

    return e.options[e.selectedIndex].value;
}

function getSelectedModelName() {
    let e = document.getElementById("mod_list");

    return e.options[e.selectedIndex].value;
}

function getSelectedExplanationName() {
    let e = document.getElementById("exp_list");

    return e.options[e.selectedIndex].value;
}

function getSelectedImageName() {
    let e = document.getElementById("img_name");

    return e.innerHTML;
}

function useNamedImage() {
    let e = document.getElementById("img_input_name");
    let imageName = e.value;

    usePredefinedImage(imageName);
}

function requestImage(url) {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsImg = JSON.parse(xmlHttp.responseText);

                let eDi = document.getElementById("div_image");
                let eIn = document.getElementById("img_name");
                let eIi = document.getElementById("img_image");
                let eIg = document.getElementById("img_groundtruth");
                let eDe = document.getElementById("div_exps");

                eDi.style.display = "block";
                eIn.innerHTML = jsImg.image_name;
                eIi.innerHTML = "<img alt='" + jsImg.image_name + "' src='data:img/jpg;base64," + jsImg.input + "'/>";
                eIg.innerHTML = jsImg.ground_truth;
                eDe.style.display = "block";
            } else {
                alert("The image failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function explain() {
    let xmlHttp = new XMLHttpRequest();
    let tgtMsg = document.getElementById("in_progress");
    let tgtBut = document.getElementById("exp_button");
    let tgtTbl = document.getElementById("exp_table");
    let ts = Date.now();
    let url = "";

    url += "/explanations-explain?";
    url += "type=json";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();
    url += "&explanation=" + getSelectedExplanationName();
    url += "&image=" + getSelectedImageName() ;

    tgtMsg.style.display = "block";
    tgtBut.style.display = "none";

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsExp = JSON.parse(xmlHttp.responseText);
                console.log(jsExp);

                let row = tgtTbl.insertRow(1);
                let cell1 = row.insertCell(0);
                let cell2 = row.insertCell(1);
                let cell3 = row.insertCell(2);
                let cell4 = row.insertCell(3);
                let cell5 = row.insertCell(4);
                let cell6 = row.insertCell(5);

                cell1.innerHTML = getSelectedExplanationName();
                cell2.innerHTML = formattedDateTime();
                cell3.innerHTML = Date.now() - ts;
                cell4.innerHTML = jsExp.explanation.prediction;
                cell5.innerHTML = jsExp.explanation.explanation_text;
                cell6.innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation.explanation_image + "'>";

                tgtMsg.style.display = "none";
                tgtBut.style.display = "block";
            } else {
                tgtMsg.style.display = "none";
                tgtBut.style.display = "block";

                alert("The explanation failed - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function predict() {
    let xmlHttp = new XMLHttpRequest();
    let tgtMsg = document.getElementById("pred_result");
    let url = "";

    url += "/models-predict?";
    url += "type=json";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();
    url += "&image=" + getSelectedImageName() ;

    tgtMsg.innerHTML = "Prediction in progress...";

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsExp = JSON.parse(xmlHttp.responseText);
                console.log(jsExp);

                tgtMsg.innerHTML = jsExp.explanation.prediction;
            } else {
                alert("The predict request failed - see server logs for details");
                tgtMsg.innerHTML = "Prediction failed";
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function formattedDateTime() {
    var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var today = new Date();
    var day = today.getDate();

    var mon = today.getMonth();
    var year = today.getFullYear();
    var hour = today.getHours();
    var min = today.getMinutes();
    var sec = today.getSeconds();

    if (day < 10) {
        day = "0" + day;
    }

    return day + "-" + months[mon] + "-" + year + " " + hour + ":" + min + ":" + sec;
}