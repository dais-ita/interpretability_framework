import {Table, Button, Header} from "semantic-ui-react";
import React, { Component } from "react";

class ModelDescription extends Component  {
    constructor(props) {
        super(props);
    }

    state = {
        colour: "grey",
        dataset_id: 0
    };

    componentDidUpdate(prevProps) {

        if (prevProps.options !== this.props.options) {
            // model updates
            if (this.props.options.model.includes(this.props.model_data.model_name)) {
                this.setState({colour: "green"})
            } else {
                this.setState({colour: "grey"})
            }


            const trained_on = this.props.model_data.trained_on;
            for (let key in trained_on) {
                if (trained_on[key]['dataset_name'] === this.props.options.dataset) {
                    this.setState({dataset_id: key})
                }
            }





        }
    }


    render () {

        return (
                <Table.Row>

                    <Table.Cell><Header as='h3'>{this.props.model_data.class_name}</Header></Table.Cell>
                    <Table.Cell>{this.props.model_data.description}</Table.Cell>
                    <Table.Cell>
                        Training Time: {this.props.model_data.trained_on[this.state.dataset_id].training_time}
                        <hr/>
                        Test Accuracy: {this.props.model_data.trained_on[this.state.dataset_id].test_accuracy}
                        <hr/>
                        Comments:
                    </Table.Cell>
                    <Table.Cell>
                        <Button color={this.state.colour} onClick={ () => this.props.setActiveModel(this.props.model_data.model_name)}>
                            Use Model
                        </Button>
                    </Table.Cell>
                </Table.Row>


            )

    }

}

export default ModelDescription



