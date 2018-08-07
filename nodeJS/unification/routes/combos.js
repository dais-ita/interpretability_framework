// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();

/* GET combos page. */
router.get('/', function(req, res, next) {
    res.render("combos", {
        "title": "P5 demo - Working combos"
    });
});

module.exports = router;
