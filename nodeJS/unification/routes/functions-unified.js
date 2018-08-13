// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

function urlPreamble(config) {
    let url =
        config.unified_apis.base.protocol +
        config.unified_apis.base.server + ":" +
        config.unified_apis.base.port + "/";

    return url;
}

module.exports = {
    datasetsAllUrl: function(config) {
        let url = urlPreamble(config) +
            config.unified_apis.dataset.all.url;

        return url;
    }
};