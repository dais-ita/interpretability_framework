// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');

/* GET api-tester page. */
router.get('/', function(req, res, next) {
    res.render("api-tester", {
        "title": "P5 demo - API Tester"
    });
});

module.exports = router;
