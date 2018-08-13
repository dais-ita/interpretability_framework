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

router.get('/', function (req, res) {
    parmDs = req.query.dataset;

    getAllDatasets(req, res, parmDs);
});

function getAllDatasets(req, res, parmDs) {
    if (parmDs != null) {
        const options = {
            method: 'GET',
            uri: fn.getDatasetsAllUrl(config)
        };

        request(options)
            .then(function (response) {
                // Success
                let dsJson = JSON.parse(response).datasets;
                let matchedDs = fn.matchedDataset(parmDs, dsJson);

                if (matchedDs != null) {
                    let dsFolder = matchedDs.folder;

                    getDatasetArchive(req, res, dsFolder);
                } else {
                    res.send("Dataset " + parmDs + " could not be located");
                }
            })
            .catch(function (err) {
                // Error
                console.log(err);
            })
    } else {
        let errMsg = "Error: No dataset specified";
        return res.status(500).send(errMsg);
    }
}

function getDatasetArchive(req, res, dsFolder) {
    const options = {
        method: 'GET',
        uri: fn.getDatasetArchiveUrl(config, dsFolder),
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
