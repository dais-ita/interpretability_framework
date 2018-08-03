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
    httpImageJson: function(config, fn, request, parmDsName, parmModName, parmExpName, parmImgName, dsJson, modJson, expJson, tgtFn) {
        if (parmDsName == null) {
            console.log("No dataset specified");
            return {};
        } else {
            const options = {
                method: 'GET'
            };

            if (parmImgName == null) {
                options.uri = fn.getDatasetsRandomTestImageUrl(config, parmDsName);
            } else {
                options.uri = fn.getDatasetsSpecificTestImageUrl(config, parmDsName, parmImgName);
            }

            request(options)
                .then(function (response) {
                    // Success
                    let result = JSON.parse(response);

                    tgtFn(config, fn, request, parmDsName, parmModName, parmExpName, dsJson, modJson, expJson, result);
                })
                .catch(function (err) {
                    // Error
                    console.log(err);
                    return {};
                })
        }
    },
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
    },
    getExplanationsForFilterUrl: function(config, dsName, modName) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.explanations.port +
            config.urls.explanations.paths.root +
            config.urls.explanations.paths.for_filters +
            "/" + dsName + "," + modName;

        return url;
    },
    getExplanationsExplainUrl: function(config) {
        let url = config.urls.base.protocol +
            config.urls.base.server + ":" +
            config.urls.explanations.port +
            config.urls.explanations.paths.root +
            config.urls.explanations.paths.explain;

        return url;
    },
    matchedDataset: function(dsName, dsJson) {
        let matchedDs = null;

        for (let i in dsJson) {
            let thisDs = dsJson[i];

            if (thisDs.dataset_name == dsName) {
                matchedDs = thisDs;
            }
        }

        return matchedDs;
    },
    matchedModel: function(modName, modJson) {
        let matchedModel = null;

        for (let i in modJson) {
            let thisMod = modJson[i];

            if (thisMod.model_name == modName) {
                matchedModel = thisMod;
            }
        }

        return matchedModel;
    },
    matchedExplanation: function(expName, expJson) {
        let matchedExp = null;

        for (let i in expJson) {
            let thisExp = expJson[i];

            if (thisExp.explanation_name == expName) {
                matchedExp = thisExp;
            }
        }

        return matchedExp;
    }
};