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
    let parmDs = req.query.dataset;
    let parmMod = req.query.model;
    let parmExp = req.query.explanation;
    let parmImg = req.query.image;

    getAllDatasets(res, parmDs, parmMod, parmExp, parmImg);
});

function getAllDatasets(res, parmDs, parmMod, parmExp, parmImg) {
    const options = {
        method: 'GET',
        uri: fn.getDatasetsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let datasets = JSON.parse(response);

            getAllModels(res, datasets, parmDs, parmMod, parmExp, parmImg);
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function getAllModels(res, datasets, parmDs, parmMod, parmExp, parmImg) {
    const options = {
        method: 'GET',
        uri: fn.getModelsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let models = JSON.parse(response);

            getAllExplanations(res, datasets, models, parmDs, parmMod, parmExp, parmImg);
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function getAllExplanations(res, datasets, models, parmDs, parmMod, parmExp, parmImg) {
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

            prepareAndExecuteExplain(res, result, parmDs, parmMod, parmExp, parmImg)
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function prepareAndExecuteExplain(res, allJson, parmDs, parmMod, parmExp, parmImg) {
    let result = {};
    let dsJson = fn.matchedDataset(parmDs, allJson.datasets);
    let modJson = fn.matchedModel(parmMod, allJson.models);
    let expJson = fn.matchedExplanation(parmExp, allJson.explanations);

    executeExplain(res, dsJson, modJson, expJson, parmDs, parmMod, parmExp, parmImg);
}

function executeExplain(res, dsJson, modJson, expJson, parmDs, parmMod, parmExp, parmImg) {
    fn.httpImageJson(config, fn, request, parmDs, parmMod, parmExp, parmImg, dsJson, modJson, expJson, function(config, fn, request, parmDs, parmMod, parmExp, dsJson, modJson, expJson, imgJson) {
        const options = {
            method: 'POST',
            uri: fn.getExplanationsExplainUrl(config),
            body: {
                "selected_dataset_json":
                    JSON.stringify(dsJson),
                "selected_model_json":
                    JSON.stringify(modJson),
                "selected_explanation_json":
                    JSON.stringify(expJson),
                "input": imgJson.input
            },
            json: true
        };

        request(options)
            .then(function (response) {
                let result = {
                    "title": "Explanations-explain",
                    "explanation": response,
                    "parameters": {
                        "dataset": parmDs,
                        "model": parmMod,
                        "explanation": parmExp,
                        "image": parmImg
                    }
                };

                result.explanation.explanation_time = new Date;
                res.json(result);
            })
            .catch(function (err) {
                // Error
                console.log(err);
                return res.sendStatus(500);
            })
    })
}

module.exports = router;
