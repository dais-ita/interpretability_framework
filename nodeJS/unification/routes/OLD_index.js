// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
    res.render("OLD_index", {
        "title": 'P5 demo'
//        "chosen_dataset": req.session.chosen_dataset,
//        "chosen_model": req.session.chosen_model,
//        "chosen_explanation": req.session.chosen_explanation
    });
});

module.exports = router;
