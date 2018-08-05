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
let parmImgName = null;

router.get('/', function (req, res) {
    parmType = req.query.type;
    parmDsName = req.query.dataset;
    parmImgName = req.query.image;

    if (parmDsName == null) {
        console.log("No dataset specified");
        return res.sendStatus(500);
    } else {
        const options = {
            method: 'GET'
        };

        if (parmImgName == null) {
            options.uri = fn.getDatasetRandomTestImageUrl(config, parmDsName);
        } else {
            options.uri = fn.getDatasetSpecificTestImageUrl(config, parmDsName, parmImgName);
        }

        request(options)
            .then(function (response) {
                // Success
                let result = JSON.parse(response);

                if (parmType != "json") {
                    let jsPage = {
                        "title": config.unified_apis.dataset.test_image.url,
                        "image": result,
                        "parameters": {
                            "type": parmType,
                            "dataset": parmDsName,
                            "image": parmImgName
                        }
                    };

                    res.render(config.unified_apis.dataset.test_image.route, jsPage);
                } else {
                    res.json(result);
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