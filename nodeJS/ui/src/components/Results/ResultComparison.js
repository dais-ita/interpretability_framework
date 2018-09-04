import React, { Component } from 'react';
import { Header, Image, Table, Icon, Segment, Button, Grid, Modal, Card} from "semantic-ui-react";
import _ from "lodash"
import axios from "axios";



import ResultQueue from "./ResultQueue";
import '../Interpreter/Interpret.css';
import ChooseInputImage from "./ChooseInputImage";


class ResultComparison extends Component {
    constructor(props) {
        super(props);
        this.state = {
            results: [],
            images: [],
            image: false
        };

        this.getResults = this.getResults.bind(this);
        this.setImage = this.setImage.bind(this);
    }

    componentDidMount () {
        // console.log(this.props.options);

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
                    <Image size='medium' src="http://placehold.it/200x200"/>
                    <Card.Content>
                        <Card.Header>Image Name</Card.Header>
                        <Card.Meta>Image Class</Card.Meta>
                    </Card.Content>
                </Card>
            )
        }


        return (
            <div>
                <Header as='h2'>&nbsp;Image Selection</Header>
                {input_image}

                <ChooseInputImage setInputImage={this.setImage} dataset={this.props.options.dataset}/>

                <Header as='h2'>Explanation Results</Header>
                <Button onClick={this.getResults}>Generate Explanation</Button>
                <br/>
                <br/>
                <ResultQueue results={this.state.results} />

            </div>

        );
    }
}

export default ResultComparison
