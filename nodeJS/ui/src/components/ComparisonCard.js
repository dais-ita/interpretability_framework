import {Header, Icon, Image, Segment} from "semantic-ui-react";
import { Table } from "semantic-ui-react";
import input_1 from "./lime_input.jpg";
import output_1 from "./lime_out.jpg";
import React, { Component } from "react";

class ComparisonCard extends Component {
    render () {
        return (

            <div>
                <Segment padded compact>
                    <Header as='h3'>{this.props.options.model[0]} trained on {this.props.options.dataset} dataset explained by {this.props.options.interpreter[0]} &nbsp;<a><Icon name='linkify'/></a> &nbsp;</Header>
                    {/*<strong>Timestamp: </strong> 17-Aug-2018 12:2:27 <br/><br/>*/}
                    {/*<strong>Model Prediction:</strong>   Gun Wielder &nbsp; &nbsp;*/}
                    {/*<strong>Ground Truth:</strong> Gun Wielder &nbsp; &nbsp;  <Icon color='green' name='thumbs up' />*/}

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
                                    {/*<Image size='small' src={input_1}/>*/}
                                    {this.props.result_data.input_image}
                                </Table.Cell>
                                <Table.Cell>
                                    {/*<Image size='small' src={output_1}/>*/}
                                    {this.props.result_data.interpretation}
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    {this.props.result_data.explanation_text}
                                    {/*<p>Evidence towards predicted class shown in green</p>*/}
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    Duration (ms): 14554
                                    {this.props.result_data.duration}
                                </Table.Cell>
                            </Table.Row>
                        </Table.Body>
                    </Table>
                </Segment>
            </div>
        )

    }
}

export default ComparisonCard
