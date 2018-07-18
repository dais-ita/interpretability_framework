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

router.get('/', function (req, res) {
    parmType = req.query.type;

    const options = {
        method: 'GET',
        uri: fn.getModelsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let result = JSON.parse(response);

            req.session.all_models = result;

            if (parmType != "json") {
                res.render("model-list", {
                    "title": "Models - all",
                    "models": result,
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
