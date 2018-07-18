// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');
let parmType = null;
let parmDs = null;
let parmMod = null;
let parmExp = null;
let parmImg = null;

router.get('/', function (req, res) {
    let dsJson = null;
    let modJson = null;
    let expJson = null;

    parmType = req.query.type;
    parmDs = req.query.dataset;
    parmMod = req.query.model;
    parmExp = req.query.explanation;
    parmImg = req.query.image;

    if (parmDs != null) {
        dsJson = fn.matchedDataset(parmDs, req.session.all_datasets);
    }

    if (parmDs != null) {
        modJson = fn.matchedModel(parmMod, req.session.all_models);
    }

    if (parmExp != null) {
        expJson = fn.matchedExplanation(parmExp, req.session.all_explanations);
    }

    if (parmImg != null) {
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
                        },
                        "chosen_dataset": req.session.chosen_dataset,
                        "chosen_model": req.session.chosen_model,
                        "chosen_explanation": req.session.chosen_explanation
                    };

                    result.explanation.explanation_time = new Date;

                    if (parmType != "json") {
                        res.render("explanations-explain", result);
                    } else {
                        res.json(result);
                    }
                })
                .catch(function (err) {
                    // Error
                    console.log(err);
                    return res.sendStatus(500);
                })
        })
    }
});

module.exports = router;
