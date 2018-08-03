// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let parmDs = null;

router.get('/', function (req, res) {
    parmDs = req.query.dataset;

    if (parmDs == "none") {
        req.session.chosen_dataset = null;
    } else {
        req.session.chosen_dataset = parmDs;
    }

    res.render("index", {
        "title": "P5 demo"
    });
});

module.exports = router;
