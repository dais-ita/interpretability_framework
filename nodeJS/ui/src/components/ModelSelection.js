import React, { Component } from 'react';
import { Header, Table } from 'semantic-ui-react';
import axios from 'axios';
import _ from 'lodash';

import ModelDescription from "./ModelDescription";
import ModelPreview from "./ModelPreview";


class ModelSelection extends Component {
    constructor(props) {
        super(props);
        this.state = {
            models : []
            // active_models : []
        }
        // this.toggleActiveModels = this.toggleActiveModels.bind(this);

    }

    componentDidMount() {
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
                    "/models-all";
        axios.get(req)
            .then(res => {
                const models = res.data;
                this.setState( { models });
                this.setState( {active_models: models[0].class_name});
            })
            .catch(function (error) {
                console.log("server error: write code to fix this! ")
            })

    }

    render () {
        const model_selections = _.times(this.state.models.length, i => (
                <React.Fragment key={i}>
                    <ModelDescription model_data={this.state.models[i]}
                                      options={this.props.options}
                                      setActiveModel={this.props.setActiveModel}/>
                    {/*<ModelPreview />*/}
                </React.Fragment>
        ));

        return (
            <div>
                <Header as='h2'>2. Machine Learning Model</Header>
                <Table basic='very' structured fixed>
                    <Table.Header>
                        <Table.Row>
                            <Table.HeaderCell colSpan={1} width={2}>Model Name</Table.HeaderCell>
                            <Table.HeaderCell colSpan={1} width={3}>Description</Table.HeaderCell>
                            <Table.HeaderCell colSpan={1} width={4}>Performance Notes</Table.HeaderCell>
                            <Table.HeaderCell colSpan={1} width={2}>&nbsp;</Table.HeaderCell>
                        </Table.Row>
                    </Table.Header>
                    <Table.Body>
                        {model_selections}

                    </Table.Body>
                </Table>

            </div>

        );


    }
}

export default ModelSelection

