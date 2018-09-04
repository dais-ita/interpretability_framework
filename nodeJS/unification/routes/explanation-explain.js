// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');

router.get('/', function (req, res) {
    let parmType = req.query.type;
    let parmDs = req.query.dataset;
    let parmMod = req.query.model;
    let parmExp = req.query.explanation;
    let parmImg = req.query.image;
    let parmMap = req.query.attribution_map;

    if (parmDs != null) {
        if (parmMod != null) {
            if (parmExp != null) {
                getAllDatasets(res, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
            } else {
                let errMsg = "Error: No explanation specified";
                return res.status(500).send(errMsg);
            }
        } else {
            let errMsg = "Error: No model specified";
            return res.status(500).send(errMsg);
        }
    } else {
        let errMsg = "Error: No dataset specified";
        return res.status(500).send(errMsg);
    }
});

function getAllDatasets(res, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    const options = {
        method: 'GET',
        uri: fn.getDatasetsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let datasets = JSON.parse(response);

            getAllModels(res, datasets, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function getAllModels(res, datasets, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    const options = {
        method: 'GET',
        uri: fn.getModelsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let models = JSON.parse(response);

            getAllExplanations(res, datasets, models, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function getAllExplanations(res, datasets, models, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    const options = {
        method: 'GET',
        uri: fn.getExplanationsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let result = {};
            let explanations = JSON.parse(response);

            result.datasets = datasets.datasets;
            result.models = models.models;
            result.explanations = explanations.explanations;

            prepareAndExecuteExplain(res, result, parmType, parmDs, parmMod, parmExp, parmImg, parmMap)
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function prepareAndExecuteExplain(res, allJson, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    let dsJson = fn.matchedDataset(parmDs, allJson.datasets);
    let modJson = fn.matchedModel(parmMod, allJson.models);
    let expJson = fn.matchedExplanation(parmExp, allJson.explanations);

    executeExplain(res, dsJson, modJson, expJson, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
}

function executeExplain(res, dsJson, modJson, expJson, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    fn.httpImageJson(config, fn, request, parmDs, parmMod, parmExp, parmImg, dsJson, modJson, expJson, function(config, fn, request, parmDs, parmMod, parmExp, parmImg, dsJson, modJson, expJson, imgJson) {
        const options = {
            method: 'POST',
            uri: fn.getExplanationExplainUrl(config),
            body: {
                "selected_dataset_json":
                    JSON.stringify(dsJson),
                "selected_model_json":
                    JSON.stringify(modJson),
                "selected_explanation_json":
                    JSON.stringify(expJson),
                "input": imgJson.input,
                "input_name": parmImg
            },
            json: true
        };

        request(options)
            .then(function (response) {
                response.explanation_time = new Date;

                if (parmMap != "false") {
                    executeAttributionMap(res, response, imgJson, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
                } else {
                    returnExplanation(res, response, imgJson, null, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
                }
            })
            .catch(function (err) {
                // Error
                console.log(err);
                return res.sendStatus(500);
            })
    })
}

function executeAttributionMap(res, expJson, imgJson, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    const options = {
        method: 'POST',
        uri: fn.getExplanationAttributionMapUrl(config),
        body: {
            "attribution_slices":
                JSON.stringify(expJson.additional_outputs.attribution_slices),
            "attribution_slice_weights":
                JSON.stringify(expJson.additional_outputs.attribution_slice_weights)
        },
        json: true
    };

    request(options)
        .then(function (response) {
            expJson.attribution_time = new Date;

            returnExplanation(res, expJson, imgJson, response, parmType, parmDs, parmMod, parmExp, parmImg, parmMap);
        })
        .catch(function (err) {
            // Error
            console.log(err);
            return res.sendStatus(500);
        })
}

function returnExplanation(res, expJson, imgJson, attJson, parmType, parmDs, parmMod, parmExp, parmImg, parmMap) {
    if (attJson != null) {
        expJson.attribution_map = attJson.attribution_map;
    }

    if (parmType == "html") {
        let jsPage = {
            "title": config.unified_apis.explanation.explain.url,
            "explanation": expJson,
            "parameters": {
                "type": parmType,
                "dataset": parmDs,
                "model": parmMod,
                "explanation": parmExp,
                "image": parmImg,
                "(chosen_image_name)": imgJson.image_name,
                "(chosen_image)": imgJson.input,
                "attribution_map": parmMap
            }
        };

        res.render(config.unified_apis.explanation.explain.route, jsPage);
    } else {
        res.json(expJson);
    }
}

module.exports = router;
