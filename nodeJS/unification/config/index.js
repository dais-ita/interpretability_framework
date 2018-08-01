let config = {};

config.datasets = {
    "hostname": "0.0.0.0",
    "port": "6001",
    "path": "/datasets/get_available",
};

config.encoding = 'utf8';

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
            "test_image_specific": "/test_image/specific"
        }
    },
    "models": {
        "port": "6101",
        "paths": {
            "root": "/models",
            "list": "/get_available",
            "for_dataset": "/get_available_for_dataset",
            "predict": "/predict"
        }
    },
    "explanations": {
        "port": "6201",
        "paths": {
            "root": "/explanations",
            "list": "/get_available",
            "for_filters": "/get_available_for_filters",
            "explain": "/explain"
        }
    }
};

module.exports = config;
