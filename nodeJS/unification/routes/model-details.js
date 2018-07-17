// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
var request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');
let parmType = null;
let parmModel = null;

router.get('/', function (req, res) {
    parmType = req.query.type;
    parmModel = req.query.model;

    const options = {
        method: 'GET',
        uri: fn.getModelsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let result = JSON.parse(response);
            let matchedModel = null;

            for (let i in result.models) {
                let thisMod = result.models[i];

                if (thisMod.model_name == parmModel) {
                    matchedModel = thisMod;
                }
            }

            if (matchedModel == null) {
                console.log("Error - no model matches '" + parmModel + "'");
            }

            if (parmType != "json") {
                res.render("model-individual", {
                    "title": "Model details",
                    "model": matchedModel,
                    "parameters": {},
                    "chosen_dataset": req.session.chosen_dataset,
                    "chosen_model": req.session.chosen_model,
                    "chosen_explanation": req.session.chosen_explanation
                });
            } else {
                res.json(result);
            }
        })
        .catch(function (err) {
            // Error
            console.log(err);
            return res.sendStatus(500);
        })
});

module.exports = router;
