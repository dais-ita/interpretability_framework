// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let parmMod = null;

router.get('/', function (req, res) {
    parmMod = req.query.model;

    if (parmMod == "none") {
        req.session.chosen_model = null;
    } else {
        req.session.chosen_model = parmMod;
    }

    res.render("index", {
        "title": "P5 demo",
        "chosen_dataset": req.session.chosen_dataset,
        "chosen_model": req.session.chosen_model,
        "chosen_explanation": req.session.chosen_explanation
    });
});

module.exports = router;
