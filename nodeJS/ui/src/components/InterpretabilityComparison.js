import React, { Component } from 'react';
import { Header, Image, Table, Icon, Segment, Button} from "semantic-ui-react";
import _ from "lodash"
import axios from "axios";

import input_1 from './lime_input.jpg';
import output_1 from './lime_out.jpg';
import ComparisonCard from "./ComparisonCard";

import './Interpret.css';
import ModelDescription from "./ModelDescription";


class InterpretabilityComparison extends Component {
    constructor(props) {
        super(props);
        this.state = {
            results: [
                {
                    input_image: "1234",
                    interpretation: "1234",
                    time_started: "123",
                    time_completed: "123",
                    explanation_text: "1234",
                    duration: "1234",
                    timestamp: "1234"
                }, {
                    input_image: "",
                    interpretation: "",
                    time_started: "",
                    time_completed: "",
                    explanation_text: "",
                    duration: "",
                    timestamp: ""
                }
            ]
        };

        this.getResults = this.getResults.bind(this);
    }

    componentDidMount () {
        console.log(this.props)



        // Query ita-ce for images
        // decode images from base-64

    }

    componentDidUpdate(prevProps) {
        // console.log(this.props);
        if (prevProps.options !== this.props.options) {
        }
    }


    getResults () {
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER;

        const rand_img = "/dataset-test-image?dataset=Gun Wielding Image Classification";

        let result = {
            input_image: "",
            interpretation: "",
            time_started: "",
            time_completed: "",
            explanation_text: "",
            duration: "",
            timestamp: ""
        };


        axios.get(req + rand_img)
            .then(res => {
                const explain_req = req + "/explanation-explain?&dataset=" +
                    encodeURIComponent(this.props.options.dataset.trim()) + "&model=" + this.props.options.model[0] +
                    "&image=" + res.data["image_name"] + "&explanation=" + this.props.options.interpreter[0];

                result["input_image"] = res.data;

                console.log(explain_req);

                axios.get(explain_req)
                    .then(res => {
                        console.log(res)
                    })
            })
    }



    render() {

        const interpreter_results = _.times(this.state.results.length, i => (
            <React.Fragment key={i}>
                <ComparisonCard result_data ={this.state.results[i]} options ={this.props.options}/>
            </React.Fragment>
        ));


        return (
            <div>
                <Button onClick={this.getResults}>Explain random image</Button>
                <Header as='h2'>Explanation Results</Header>
                {interpreter_results}
            </div>

        );
    }
}

export default InterpretabilityComparison