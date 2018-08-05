// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');

/* GET home page. */
router.get('/', function(req, res, next) {
    const options = {
        method: 'GET',
        uri: fn.getDatasetsAllUrl(config)
    };

    request(options)
        .then(function (response) {
            // Success
            let result = JSON.parse(response);
            let jsPage = {
                "title": "P5 demo",
                "datasets": result
            };

            res.render("index", jsPage);
        })
        .catch(function (err) {
            // Error
            console.log(err);
            return res.sendStatus(500);
        })
});

module.exports = router;
