import React, { Component } from 'react';
import { Header, Image, Table, Icon, Segment, Button, Grid, Modal, Card} from "semantic-ui-react";
import _ from "lodash"
import axios from "axios";

//import input_1 from './lime_input.jpg';
//import output_1 from './lime_out.jpg';
import ComparisonCard from "./ComparisonCard";

import './Interpret.css';
import ModelDescription from "./ModelDescription";


class InterpretabilityComparison extends Component {
    constructor(props) {
        super(props);
        this.state = {
            results: [],
            images: [],
            image: false
        };

        this.getResults = this.getResults.bind(this);
        this.loadImages = this.loadImages.bind(this);
    }

    componentDidMount () {
        console.log(this.props)

    }

    componentDidUpdate(prevProps) {
        // console.log(this.props);
        if (prevProps.options !== this.props.options) {
        }
    }


    getResults () {
        const prevResults = this.state.results;
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER;

        const img = "/dataset-test-image?dataset=Gun Wielding Image Classification&image=" + this.state.image.image_name;

        let result = {};

        result["dataset"] = this.props.options.dataset;
        result["model"] = this.props.options.model[0];
        result["interpreter"] = this.props.options.interpreter[0];

        axios.get(req + img)
            .then(res => {
                const explain_req = req + "/explanation-explain?dataset=" +
                    encodeURIComponent(this.props.options.dataset.trim()) + "&model=" + this.props.options.model[0] +
                    "&image=" + res.data["image_name"] + "&explanation=" + this.props.options.interpreter[0];

                result["input_image"] = res.data["input"];
                result["ground_truth"] = res.data["ground_truth"];
                result["image_name"] = res.data["image_name"];

                axios.get(explain_req)
                    .then(res => {
                        result["start_time"] = res.data["attribution_time"];
                        result["explanation_image"] = res.data["explanation_image"];
                        result["explanation_text"] = res.data["explanation_text"];
                        result["end_time"] = res.data["explanation_time"];
                        result["prediction"] = res.data["prediction"];
                    })
                    .catch( function(error) {
                        alert(error);
                    })



            });

        prevResults.push(result);
        this.setState({results: prevResults});

        console.log(this.state.results);

    }

    showPreview(result_id) {
         alert(result_id);
    }

    loadImages() {
        axios.get("http://127.0.0.1:3100/dataset-test-images?dataset=Gun%20Wielding%20Image%20Classification&num_images=100")
            .then(res => {
                this.setState({images : res.data})
            });
    }


    setImage(img_data) {
        this.setState({image: img_data})
    }





    render() {

        const interpreter_results = _.times(this.state.results.length, i => (
            <Grid.Column key={i}>
                <Modal trigger={<Image src={"data:image/png;base64," + this.state.results[i].explanation_image}/>}>
                    <Modal.Header>{this.state.results[i].model} trained on {this.state.results[i].dataset} dataset explained by
                        {this.state.results[i].interpreter}</Modal.Header>
                    <Modal.Content>
                    {/*<Segment padded compact>*/}
                        {/*<Header as='h3'>{this.state.data.model} trained on {this.state.data.dataset} dataset explained by {this.state.data.interpreter} &nbsp;<a><Icon name='down arrow'/></a> &nbsp;</Header>*/}
                        {/*<strong>Timestamp: </strong> 17-Aug-2018 12:2:27 <br/><br/>*/}
                        {/*<strong>Model Prediction:</strong>   Gun Wielder &nbsp; &nbsp;*/}
                        {/*<strong>Ground Truth:</strong> Gun Wielder &nbsp; &nbsp;  <Icon color='green' name='thumbs up' />*/}

                        <Table basic='very' collapsing>
                            <Table.Header>
                                <Table.Row>
                                    <Table.HeaderCell>Input Image</Table.HeaderCell>
                                    <Table.HeaderCell>Explanation Image</Table.HeaderCell>
                                    <Table.HeaderCell>Explanation Text</Table.HeaderCell>
                                    <Table.HeaderCell>Further Details</Table.HeaderCell>
                                </Table.Row>
                            </Table.Header>
                            <Table.Body>
                                <Table.Row>
                                    <Table.Cell>
                                        {/*<Image size='small' src={input_1}/>*/}
                                        <Image size='small' src={"data:image/png;base64," + this.state.results[i].input_image} />
                                        <p>{this.state.results[i].ground_truth}</p>
                                    </Table.Cell>
                                    <Table.Cell>
                                        <Image size='small' src={"data:image/png;base64," + this.state.results[i].explanation_image}/>
                                        <p>{this.state.results[i].prediction}</p>
                                    </Table.Cell>
                                    <Table.Cell verticalAlign='top'>
                                        {this.state.results[i].explanation_text}
                                        {/*<p>Evidence towards predicted class shown in green</p>*/}
                                    </Table.Cell>
                                    <Table.Cell verticalAlign='top'>
                                        {this.state.results[i].start_time}
                                    </Table.Cell>
                                </Table.Row>
                            </Table.Body>
                        </Table>
                    </Modal.Content>
                </Modal>
                        {/*<Modal.Content image >*/}
                                {/*<Image wrapped size='medium' src={"data:image/png;base64," + this.state.results[i].input_image} />*/}
                                {/*Input*/}
                                {/*<Image wrapped size='medium' src={"data:image/png;base64," + this.state.results[i].explanation_image} />*/}
                                {/*Output*/}
                          {/*<Modal.Description>*/}
                              {/*<Table basic='very'>*/}
                                  {/*<Table.Header>*/}
                                      {/*<Table.Row>*/}
                                          {/*/!*<Table.HeaderCell>Input Image</Table.HeaderCell>*!/*/}
                                          {/*/!*<Table.HeaderCell>Explanation Image</Table.HeaderCell>*!/*/}
                                          {/*<Table.HeaderCell>Explanation Text</Table.HeaderCell>*/}
                                          {/*<Table.HeaderCell>Further Details</Table.HeaderCell>*/}
                                      {/*</Table.Row>*/}
                                  {/*</Table.Header>*/}
                                  {/*<Table.Body>*/}
                                      {/*<Table.Row>*/}
                                          {/*<Table.Cell verticalAlign='top'>*/}
                                              {/*{this.state.results[i].explanation_text}*/}
                                          {/*</Table.Cell>*/}
                                          {/*<Table.Cell verticalAlign='top'>*/}
                                              {/*{this.state.results[i].start_time}*/}
                                          {/*</Table.Cell>*/}
                                      {/*</Table.Row>*/}
                                  {/*</Table.Body>*/}
                              {/*</Table>*/}
                            {/*/!*<Header>Default Profile Image</Header>*!/*/}
                            {/*/!*<p>We've found the following gravatar image associated with your e-mail address.</p>*!/*/}
                            {/*/!*<p>Is it okay to use this photo?</p>*!/*/}
                          {/*</Modal.Description>*/}
                        {/*</Modal.Content>*/}

            </Grid.Column>
        ));

        const dataset_images = _.times(this.state.images.length, i => (

            <Grid.Column key={i}>
                <Segment onClick={() => this.setImage(this.state.images[i])}>
                    <Image size='small' src={"data:image/png;base64," + this.state.images[i].input} />
                    {this.state.images[i].ground_truth}<br/>
                    {this.state.images[i].image_name}
                </Segment>
            </Grid.Column>

        ));

        let input_image;
        let results;

        if (this.state.results.length > 0) {
            results = interpreter_results
        } else {
            results = <br/>
        }


        if (this.state.image) {
            input_image = (
                <Segment>
                    <Image src={"data:image/png;base64," + this.state.image.input} />
                    <p>{this.state.image.ground_truth}</p>
                    <p>{this.state.image.image_name}</p>
                </Segment>
            )
        } else {
            input_image = <p>Please select input</p>
        }



        return (
            <div>
                <Header as='h2'>4. Image Selection</Header>
                <Modal onOpen={this.loadImages} trigger={<Button>Choose Image</Button>} centered={false}>
                    <Modal.Header>{this.props.options.dataset}</Modal.Header>
                    <Modal.Content>
                        <Grid stackable columns={6}>
                            {dataset_images}
                        </Grid>
                    </Modal.Content>

                </Modal>

                {input_image}

                <Header as='h2'>5. Explanation Results</Header>
                <Button onClick={this.getResults}>Generate Explanation</Button>

                <br/>
                <Grid stackable columns={4}>
                    {results}
                </Grid>

            </div>

        );
    }
}

export default InterpretabilityComparison
