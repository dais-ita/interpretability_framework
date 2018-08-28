import React, { Component } from 'react';
import { Header, Image, Grid, Table } from "semantic-ui-react";
import _ from 'lodash';
import axios from "axios";

import InterpretabilityDescription from "./InterpretabilityDescription";

class InterpretabilitySelection extends Component {
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
                console.log(res.data);
                const interpreters = res.data;
                this.setState( { interpreters });
            });

    }


    render() {

        const interpreter_descriptions = _.times(this.state.interpreters.length, i => (
            <React.Fragment key={i}>
                <InterpretabilityDescription interpreter_data={this.state.interpreters[i]}
                                             options={this.props.options}
                                             setActiveInterpreter={this.props.setActiveInterpreter}/>
            </React.Fragment>
        ));


        return (
            <div>
                <Header as='h2'>3. Interpretability Technique Selection</Header>
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

export default InterpretabilitySelection