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
let parmDsName = null;

router.get('/', function (req, res) {
    parmType = req.query.type;
    parmDsName = req.query.dataset;

    if (parmDsName == null) {
        console.log("No dataset specified");
        return res.sendStatus(500);
    } else {
        const options = {
            method: 'GET',
            uri: fn.getModelsForDatasetUrl(config, parmDsName)
        };

        request(options)
            .then(function (response) {
                // Success
                let result = JSON.parse(response);

                if (parmType == "html") {
                    let jsPage = {
                        "title": config.unified_apis.model.for_dataset.url,
                        "models": result,
                        "parameters": {
                            "type": parmType,
                            "dataset": parmDsName
                        }
                    };

                    res.render(config.unified_apis.model.for_dataset.route, jsPage);
                } else {
                    res.json(result)
                }
            })
            .catch(function (err) {
                // Error
                console.log(err);
                return res.sendStatus(500);
            })
    }
});

module.exports = router;
