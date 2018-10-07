import React, { Component } from 'react';
import { Image, Card, Label } from "semantic-ui-react";
import axios from 'axios';


class DatasetCard extends Component {
    constructor(props) {
        super(props);
        // this.toggleState = this.toggleState.bind(this);

        this.state = {
            active: false,
            name: this.props.dataset.dataset_name,
            preview_image: "",
            colour: "grey"
        };
    }

    componentDidMount () {
        // console.log("###")
        axios.get("http://" + process.env.REACT_APP_SERVER_NAME +
                  ":" + process.env.REACT_APP_PORT_NUMBER +
                  "/dataset-test-image?dataset=" +
                  this.props.dataset.dataset_name + "&image=" + this.props.dataset.interesting_images[0])
            .then(res => {
                this.setState({preview_image: "data:image/png;base64," + res.data.input})
            })

    }


    componentDidUpdate(prevProps) {

        /* Set component to active if the `active` dataset is changed and equal to this component */
        if (prevProps.active_dataset !== this.props.active_dataset) {
            if (this.props.active_dataset === this.state.name) {
                this.setState({active : true});
                this.setState({colour : "green"});
            } else {
                this.setState({active : false});
                this.setState({colour : "grey"});
            }
        }
    }


    render() {

        let label;
        if (this.state.active) {
            label = <Label color='green' ribbon="right">Selected</Label>
        } else {
            label = <br/>
        }

        return (
            <div onClick={() => this.props.setActiveDataset(this.props.dataset.dataset_name)}>
                <Card color={this.state.colour}>
                    <Image src={this.state.preview_image} size="medium"/>
                    <Card.Content>
                        {label}
                        <Card.Header>{this.props.dataset.dataset_name}</Card.Header>
                        <Card.Description>{this.props.dataset.description}</Card.Description>
                    </Card.Content>
                </Card>
            </div>
        )
    }


}


export default DatasetCard