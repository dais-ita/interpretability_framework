// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let parmExp = null;

router.get('/', function (req, res) {
    parmExp = req.query.explanation;

    if (parmExp == "none") {
        req.session.chosen_explanation = null;
    } else {
        req.session.chosen_explanation = parmExp;
    }

    res.render("index", {
        "title": "P5 demo",
        "chosen_dataset": req.session.chosen_dataset,
        "chosen_model": req.session.chosen_model,
        "chosen_explanation": req.session.chosen_explanation
    });
});

module.exports = router;
