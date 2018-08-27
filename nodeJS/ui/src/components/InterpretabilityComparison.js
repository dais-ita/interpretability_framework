import React, { Component } from 'react';
import { Header, Image, Table, Icon, Segment} from "semantic-ui-react";
import input_1 from './lime_input.jpg';
import output_1 from './lime_out.jpg';

import './Interpret.css';


class InterpretabilityComparison extends Component {

    state = {

    };

    componentDidMount () {
        // Query ita-ce for images
        // decode images from base-64

    }


    render() {


        return (
            <div>
                <Header as='h2'>Explanation Results</Header>

                <Segment padded compact>
                    <Header as='h3'>CNN_1 trained on Gun Wielding Image Classification dataset explained by LIME &nbsp;<a><Icon name='linkify'/></a> &nbsp;</Header>
                    <strong>Timestamp: </strong> 17-Aug-2018 12:2:27 <br/><br/>
                    <strong>Model Prediction:</strong>   Gun Wielder &nbsp; &nbsp;
                    <strong>Ground Truth:</strong> Gun Wielder &nbsp; &nbsp;  <Icon color='green' name='thumbs up' />

                    <Table collapsing>
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
                                    <Image size='small' src={input_1}/>
                                </Table.Cell>
                                <Table.Cell>
                                    <Image size='small' src={output_1}/>
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    <p>Evidence towards predicted class shown in green</p>
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    Duration (ms): 14554
                                </Table.Cell>
                            </Table.Row>
                        </Table.Body>
                    </Table>
                </Segment>
            </div>
        );
    }
}

export default InterpretabilityComparison