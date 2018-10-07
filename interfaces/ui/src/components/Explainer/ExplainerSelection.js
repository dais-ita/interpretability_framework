import React, { Component } from 'react';
// import { Header, Image, Grid, Table } from "semantic-ui-react";
import { Table } from "semantic-ui-react";
import _ from 'lodash';
import axios from "axios";

import ExplainerDescription from "./ExplainerDescription";

class ExplainerSelection extends Component {
    constructor(props) {
        super(props);

        this.state = {
            interpreters: []
        };

    }


    componentDidMount () {
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
            "/explanations-all";

        axios.get(req)
            .then(res => {
                const interpreters = res.data;
                this.setState( { interpreters });
                this.props.setActiveInterpreter(this.state.interpreters[0].explanation_name);
            });

    }


    render() {

        const interpreter_descriptions = _.times(this.state.interpreters.length, i => (
            <React.Fragment key={i}>
                <ExplainerDescription interpreter_data={this.state.interpreters[i]}
                                      options={this.props.options}
                                      setActiveInterpreter={this.props.setActiveInterpreter}/>
            </React.Fragment>
        ));


        return (
            <div>
                <Table basic='very' structured fixed>

                    <Table.Header>
                        <Table.Row>
                            <Table.HeaderCell width={2}>Interpretability Technique</Table.HeaderCell>
                            <Table.HeaderCell width={4}>Description</Table.HeaderCell>
                            <Table.HeaderCell width={2}>&nbsp;</Table.HeaderCell>
                        </Table.Row>
                    </Table.Header>
                    <Table.Body>
                        {interpreter_descriptions}
                    </Table.Body>
                </Table>
            </div>
        );
    }
}

export default ExplainerSelection