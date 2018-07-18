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

    let url = fn.getModelsPredictUrl(config);

    console.log("url: " + url);
    console.log("dsName: " + parmDsName);
    console.log("imgName: " + parmImgName);

    if (parmDsName == null) {
        console.log("No dataset specified");
        return res.sendStatus(500);
    } else {
        if (parmImgName == null) {
            imgName = "random";
        } else {
            imgName = parmImgName;
        }

        const options = {
            method: 'POST',
            uri: fn.getModelsPredictUrl(config, parmDsName, imgName)
        };

        request(options)
            .then(function (response) {
                // Success
                let result = JSON.parse(response);

                if (parmType != "json") {
                    res.render("models-predict", {
                        "title": "Models - predict",
                        "dataset": dsName,
                        "image": imgName,
                        "models": result,
                        "parameters": {
                            "dataset": parmDataset,
                            "image": parmImgName
                        }
//                        "chosen_dataset": req.session.chosen_dataset,
//                        "chosen_model": req.session.chosen_model,
//                        "chosen_explanation": req.session.chosen_explanation
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