// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();

/* GET readme page. */
router.get('/', function(req, res, next) {
    res.render("readme", {
        "title": "P5 demo - Help"
    });
});

module.exports = router;
