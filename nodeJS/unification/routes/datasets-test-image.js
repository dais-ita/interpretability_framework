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
            options.uri = fn.getDatasetsRandomTestImageUrl(config, parmDsName);
        } else {
            options.uri = fn.getDatasetsSpecificTestImageUrl(config, parmDsName, parmImgName);
        }

        request(options)
            .then(function (response) {
                // Success
                let result = JSON.parse(response);

                if (parmType != "json") {
                    res.render("image-details", {
                        "title": "Datasets - test image",
                        "image": result,
                        "parameters": {"dataset": parmDsName, "image": parmImgName},
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
    }
});

module.exports = router;