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
        uri: fn.getExplanationsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let result = JSON.parse(response);

            if (parmType != "json") {
                let jsPage = {
                    "title": config.unified_apis.explanation.all.url,
                    "explanations": result,
                    "parameters": {
                        "type": parmType
                    }
                };

                res.render(config.unified_apis.explanation.all.route, jsPage);
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
