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
    let parmImg = req.query.image;

    getAllDatasets(res, parmType, parmDs, parmMod, parmImg);
});

function getAllDatasets(res, parmType, parmDs, parmMod, parmImg) {
    const options = {
        method: 'GET',
        uri: fn.getDatasetsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let datasets = JSON.parse(response);

            prepareAndExecutePredict(res, datasets.datasets, parmType, parmDs, parmMod, parmImg);
        })
        .catch(function (err) {
            // Error
            console.log(err);
        })
}

function prepareAndExecutePredict(res, datasets, parmType, parmDs, parmMod, parmImg) {
    let dsJson = fn.matchedDataset(parmDs, datasets);

    executePredict(res, dsJson, parmType, parmDs, parmMod, parmImg);
}

function executePredict(res, dsJson, parmType, parmDs, parmMod, parmImg) {
    fn.httpImageJson(config, fn, request, parmDs, parmMod, null, parmImg, dsJson, null, null, function(config, fn, request, parmDs, parmMod, parmExp, parmImg, dsJson, modJson, expJson, imgJson) {
        const options = {
            method: 'POST',
            uri: fn.getModelPredictUrl(config),
            body: {
                "selected_dataset_json":
                    JSON.stringify(dsJson),
                "selected_model": parmMod,
                "input": imgJson.input
            },
            json: true
        };

        request(options)
            .then(function (response) {
                if (parmType != "json") {
                    let jsPage = {
                        "title": config.unified_apis.model.predict.url,
                        "prediction": response,
                        "parameters": {
                            "type": parmType,
                            "dataset": parmDs,
                            "model": parmMod,
                            "image": parmImg,
                            "(chosen_image_name)": imgJson.image_name,
                            "(chosen_image)": imgJson.input
                        }
                    };

                    res.render(config.unified_apis.model.predict.route, jsPage);
                } else {
                    res.json(response);
                }
            })
            .catch(function (err) {
                // Error
                console.log(err);
                return res.sendStatus(500);
            })
    })
}

module.exports = router;
