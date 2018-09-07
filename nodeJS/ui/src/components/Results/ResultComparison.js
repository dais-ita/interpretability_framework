import React, { Component } from 'react';
// import { Header, Image, Table, Icon, Segment, Button, Grid, Modal, Card, Divider} from "semantic-ui-react";
import { Header, Image, Button, Card} from "semantic-ui-react";
import axios from "axios";

import ResultQueue from "./ResultQueue";
import InputImageModal from "./InputImageModal";
import placeholder from "./no_image_selected.png";

import ResultTable from "./ResultTable";

import _ from "lodash";



class ResultComparison extends Component {
    constructor(props) {
        super(props);
        this.state = {
            results: [],
            images: [],
            image: false,
            multi_results: [],
            interpreters: []
        };

        this.getResults = this.getResults.bind(this);
        this.getMultipleResults = this.getMultipleResults.bind(this);
        this.setImage = this.setImage.bind(this);
    }

    componentDidMount () {
        axios.get("http://" + process.env.REACT_APP_SERVER_NAME + ":"
            + process.env.REACT_APP_PORT_NUMBER + "/explanations-all")
            .then(res => {
                const interpreters = res.data;
                this.setState( { interpreters })
            });
    }

    componentDidUpdate(prevProps) {

        if (prevProps.options !== this.props.options) {

            // console.log(this.state.results);
            // this.setState(this.state);
        }
    }


    getResults () {
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER;
        const img = "/dataset-test-image?dataset=" + encodeURIComponent(this.props.options.dataset.trim())
                    + "&image=" + this.state.image.image_name;

            let result = {};

            result["dataset"]      = this.props.options.dataset;
            result["model"]        = this.props.options.model;
            result["interpreter"]  = this.props.options.interpreter;

            axios.get(req + img)
                .then(res => {
                    const explain_req = req + "/explanation-explain?dataset=" +
                        encodeURIComponent(this.props.options.dataset.trim()) + "&model=" + this.props.options.model +
                        "&image=" + res.data["image_name"] + "&explanation=" + this.props.options.interpreter;

                    result["input_image"]  = res.data["input"];
                    result["ground_truth"] = res.data["ground_truth"];
                    result["image_name"]   = res.data["image_name"];

                    axios.get(explain_req)
                        .then(res => {
                            result["start_time"]        = res.data["attribution_time"];
                            result["explanation_image"] = res.data["explanation_image"];
                            result["explanation_text"]  = res.data["explanation_text"];
                            result["end_time"]          = res.data["explanation_time"];
                            result["prediction"]        = res.data["prediction"];

                            const results = this.state.results;
                            results.push(result);
                            this.setState({results});
                            console.log(this.state.results);
                        })
                        .catch( function(error) {
                            alert(error);
                        })
                });


    }

    getMultipleResults () {
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER;
        const img = "/dataset-test-image?dataset=" + encodeURIComponent(this.props.options.dataset.trim());
            // + "&image=" + this.state.image.image_name;

        let result_entry = [];

            axios.get(req + img)
                .then(res => {
                    for (let i = 0; i < this.state.interpreters.length; i++ ) {
                        let result = {};

                        result["dataset"] = this.props.options.dataset;
                        result["model"] = this.props.options.model;
                        result["interpreter"] = this.state.interpreters[i].explanation_name;

                        const explain_req = req + "/explanation-explain?dataset=" +
                            encodeURIComponent(this.props.options.dataset.trim()) + "&model=" + this.props.options.model +
                            // "&image=" + res.data["image_name"] + "&explanation=" + this.state.interpreters[i].explanation_name;
                            "&image=" + res.data["image_name"] + "&explanation=LIME";

                        result["input_image"] = res.data["input"];
                        result["ground_truth"] = res.data["ground_truth"];
                        result["image_name"] = res.data["image_name"];
                        console.log("processing");

                        axios.get(explain_req)
                            .then(res => {
                                result["start_time"] = res.data["attribution_time"];
                                result["explanation_image"] = res.data["explanation_image"];
                                result["explanation_text"] = res.data["explanation_text"];
                                result["end_time"] = res.data["explanation_time"];
                                result["prediction"] = res.data["prediction"];

                                // const results = this.state.results;
                                result_entry.push(result);
                                console.log(this.state);
                                this.setState(this.state);

                            })
                            .catch(function (error) {
                                alert(error);
                            })
                    }}
                    );

        const multi_results = this.state.multi_results;
        multi_results.push(result_entry);
        this.setState({multi_results});

    }


    setImage(img_data) {
        this.setState({image: img_data})
    }

    render() {
        let input_image;

        if (this.state.image) {
            input_image = (
                <Card>
                    <Image size='medium' src={"data:image/png;base64," + this.state.image.input}/>
                    <Card.Content>
                        <Card.Header>{this.state.image.image_name}</Card.Header>
                        <Card.Meta>{this.state.image.ground_truth}</Card.Meta>
                    </Card.Content>
                </Card>
            )
        } else {
            input_image = (
                <Card>
                    <Image size='medium' src={placeholder}/>
                    <Card.Content>
                        <Card.Header>Image Title</Card.Header>
                        <Card.Meta>Ground Truth</Card.Meta>
                    </Card.Content>
                </Card>
            )
        }

        let image_selection;


        // Logic here for multiple explainers
        let results;


        if (this.props.options.use_case === 1) {
            results = (
                <div>
                    <Button onClick={this.getMultipleResults}>Generate Explanations</Button>
                    {/*<Button onClick={_.times(50, i => (this.getResults))}>Generate 50 Random Explanations</Button>*/}
                    <ResultTable results={this.state.multi_results} />
                </div>
            );

        } else {
            results = (
                <div>
                    <Button onClick={this.getResults}>Generate Explanation</Button>
                    <ResultQueue results={this.state.results} />
                </div>
                );

            image_selection = (
                <div>
                    <Header as='h3'>&nbsp;Image Selection</Header>
                    {input_image}
                    <InputImageModal setInputImage={this.setImage} dataset={this.props.options.dataset}/>
                </div>
            )
        }



        return (
            <div>
                <br/>
                <Header as='h2'>Explanations</Header>
                {image_selection}

                <Header dividing as='h2'>Results</Header>
                {results}
            </div>

        );
    }
}

export default ResultComparison
