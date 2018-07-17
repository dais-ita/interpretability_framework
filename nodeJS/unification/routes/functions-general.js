// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

function httpGet(hostname, port, path, encoding, passedRes) {
    let http = require('http');
    let result = "";

    const options = {
        hostname: hostname,
        port: port,
        path: path,
        method: 'GET',
        headers: {
            'Content-Type': 'text/plain; charset=utf-8'
        }
    };

    const req = http.request(options, (res) => {
        res.setEncoding(encoding);
        res.on('data', (chunk) => {
            result += chunk;
        });
        res.on('end', () => {
            passedRes.json(JSON.parse(result));
        });
    });

    req.on('error', (e) => {
        passedRes.json(JSON.parse(e));
    });

    req.end();
}

function httpPost(hostname, port, path, encoding, payload, passedRes) {
    let querystring = require('querystring');
    let http = require('http');
    let encPayload = querystring.escape(payload);
    let result = "";

    const options = {
        hostname: hostname,
        port: port,
        path: path,
        method: 'POST',
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'Content-Length': Buffer.byteLength(encPayload)
        }
    };

    const req = http.request(options, (res) => {
        res.setEncoding(encoding);
        res.on('data', (chunk) => {
            result += chunk;
        });
        res.on('end', () => {
            passedRes.json(JSON.parse(result));
        });
    });

    req.on('error', (e) => {
        passedRes.json(JSON.parse(e));
    });

    req.write(encPayload);
    req.end();

}

module.exports = {
    getDatasetsAllUrl: function(config) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.datasets.port +
            config.urls.datasets.paths.root +
            config.urls.datasets.paths.list;

        return url;
    },
    getDatasetsRandomTestImageUrl: function(config, dsName) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.datasets.port +
            config.urls.datasets.paths.root +
            config.urls.datasets.paths.test_image_random +
            "/" + dsName;

        return url;
    },
    getDatasetsSpecificTestImageUrl: function(config, dsName, imgName) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.datasets.port +
            config.urls.datasets.paths.root +
            config.urls.datasets.paths.test_image_specific +
            "?dataset=" + dsName +
            "&image_name=" + imgName;

        return url;
    },
    getModelsAllUrl: function(config) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.models.port +
            config.urls.models.paths.root +
            config.urls.models.paths.list;

        return url;
    },
    getModelsForDatasetUrl: function(config, dsName) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.models.port +
            config.urls.models.paths.root +
            config.urls.models.paths.for_dataset +
            "/" + dsName;

        return url;
    },
    getModelsPredictUrl: function(config) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.models.port +
            config.urls.models.paths.root +
            config.urls.models.paths.predict;

        return url;
    },
    getExplanationsAllUrl: function(config) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.explanations.port +
            config.urls.explanations.paths.root +
            config.urls.explanations.paths.list;

        return url;
    }
};