let config = {};

config.encoding = 'utf8';

config.unified_apis = {
    "base": {
        "protocol": "http://",
        "server": "0.0.0.0",
        "port": "3100"
    },
    "dataset": {
        "all": {
            "url": "datasets-all",
            "route": "dataset-list",
            "maps_to": "config.urls.datasets.list"
        },
        "details": {
            "url": "dataset-details",
            "route": "dataset-individual",
            "maps_to": null
        },
        "test_image": {
            "url": "dataset-test-image",
            "route": "image-individual",
            "maps_to": [ "config.urls.datasets.test_image_random", "config.urls.datasets.test_image_specific" ]
        },
        "test_images": {
            "url": "dataset-test-images",
            "route": "image-list",
            "maps_to": "config.urls.datasets.image_list"
        },
        "archive": {
            "url": "archive",
            "route": null,
            "maps_to": "config.urls.datasets.archive"
        }
    },
    "model": {
        "all": {
            "url": "models-all",
            "route": "model-list",
            "maps_to": "config.urls.models.list"
        },
        "details": {
            "url": "model-details",
            "route": "model-individual",
            "maps_to": null
        },
        "for_dataset": {
            "url": "models-for-dataset",
            "route": "model-list",
            "maps_to": "config.urls.models.for_dataset"
        },
        "predict": {
            "url": "model-predict",
            "route": "model-predict",
            "maps_to": "config.urls.models.predict"
        },
        "archive": {
            "url": "archive",
            "route": null,
            "maps_to": "config.urls.models.archive"
        }
    },
    "explanation": {
        "all": {
            "url": "explanations-all",
            "route": "explanation-list",
            "maps_to": "config.urls.explanations.list"
        },
        "details": {
            "url": "explanation-details",
            "route": "explanation-individual",
            "maps_to": null
        },
        "for_filter": {
            "url": "explanations-for-filter",
            "route": "explanation-list",
            "maps_to": "config.urls.explanations.for_filters"
        },
        "explain": {
            "url": "explanation-explain",
            "route": "explanation-explain",
            "maps_to": "config.urls.explanations.explain"
        },
        "attribution_map": {
            "url": "explanation-attribution-map",
            "route": "explanation-attribution-map",
            "maps_to": "config.urls.explanations.attribution_map"
        }
    }
};

config.urls = {
    "base": {
        "protocol": "http://",
        "server": "0.0.0.0"
    },
    "datasets": {
        "port": "6001",
        "paths": {
            "root": "/datasets",
            "list": "/get_available",
            "test_image_random": "/test_image",
            "test_image_specific": "/test_image/specific",
            "image_list": "/test_images",
            "archive": "/archive"
        }
    },
    "models": {
        "port": "6101",
        "paths": {
            "root": "/models",
            "list": "/get_available",
            "for_dataset": "/get_available_for_dataset",
            "predict": "/predict",
            "archive": "/archive"
        }
    },
    "explanations": {
        "port": "6201",
        "paths": {
            "root": "/explanations",
            "list": "/get_available",
            "for_filters": "/get_available_for_filters",
            "explain": "/explain",
            "attribution_map": "/attribution_map"
        }
    }
};

module.exports = config;
