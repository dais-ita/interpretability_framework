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
let parmExp = null;

router.get('/', function (req, res) {
    parmType = req.query.type;
    parmExp = req.query.explanation;

    const options = {
        method: 'GET',
        uri: fn.getExplanationsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let result = JSON.parse(response).explanations;
            let matchedExp = fn.matchedExplanation(parmExp, result);

            if (matchedExp == null) {
                console.log("Error - no explanation matches '" + parmExp + "'");
            }

            if (parmType != "json") {
                let jsPage =  {
                    "title": "Explanation details",
                    "explanation": matchedExp,
                    "parameters": {
                        "type": parmType,
                        "explanation": parmExp
                    }
                };

                res.render("explanation-individual", jsPage);
            } else {
                res.json(matchedExp);
            }
        })
        .catch(function (err) {
            // Error
            console.log(err);
            return res.sendStatus(500);
        })
});

module.exports = router;
