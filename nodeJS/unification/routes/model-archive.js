// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');
let parmDs = null;
let parmMod = null;

router.get('/', function (req, res) {
    parmDs = req.query.dataset;
    parmMod = req.query.model;

    if (parmDs != null) {
        if (parmMod != null) {
            if (parmMod != "cnn_1") {
                getModelArchive(req, res, parmDs, parmMod);
            } else {
                let errMsg = "The " + parmMod + " model is too large to be returned - please obtain manually";
                return res.status(500).send(errMsg);
            }
        } else {
            let errMsg = "Error: No model specified";
            return res.status(500).send(errMsg);
        }
    } else {
        let errMsg = "Error: No dataset specified";
        return res.status(500).send(errMsg);
    }
});

function getModelArchive(req, res, parmDs, parmMod) {
    let lcParmDs = parmDs.toLowerCase();      // This should be handled server side, not here
    const options = {
        method: 'GET',
        uri: fn.getModelArchiveUrl(config, lcParmDs, parmMod),
        resolveWithFullResponse: true,
        encoding: null
    };

    request(options)
        .then(function (response) {
            // Success
            let data = response.body;

            res.writeHead(200, {
                'Content-Type': response.headers["content-type"],
                'Content-disposition': response.headers["content-disposition"],
                'Content-Length': data.length
            });
            res.end(data);
        })
        .catch(function (err) {
            // Error
            console.log(err);
            return res.sendStatus(500);
        })
}

module.exports = router;
