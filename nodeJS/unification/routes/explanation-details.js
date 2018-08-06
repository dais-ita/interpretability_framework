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

    if (parmExp != null) {
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

                if (parmType == "html") {
                    let jsPage =  {
                        "title": config.unified_apis.explanation.details.url,
                        "explanation": matchedExp,
                        "parameters": {
                            "type": parmType,
                            "explanation": parmExp
                        }
                    };

                    res.render(config.unified_apis.explanation.details.route, jsPage);
                } else {
                    res.json(matchedExp);
                }
            })
            .catch(function (err) {
                // Error
                console.log(err);
                return res.sendStatus(500);
            })
    } else {
        let errMsg = "Error: No explanation specified";
        return res.status(500).send(errMsg);
    }
});

module.exports = router;
