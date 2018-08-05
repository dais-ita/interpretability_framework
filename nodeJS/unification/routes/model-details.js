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
            let result = JSON.parse(response).models;
            let matchedModel = fn.matchedModel(parmModel, result);

            if (matchedModel == null) {
                console.log("Error - no model matches '" + parmModel + "'");
            }

            if (parmType != "json") {
                let jsPage = {
                    "title": "Model details",
                    "model": matchedModel,
                    "parameters": {
                        "type": parmType,
                        "model": parmModName
                    }
                };

                res.render("model-individual", jsPage);
            } else {
                res.json(matchedModel);
            }
        })
        .catch(function (err) {
            // Error
            console.log(err);
            return res.sendStatus(500);
        })
});

module.exports = router;
