import {Image, Modal, Table} from "semantic-ui-react";
import React, { Component } from "react";


class ResultModal extends Component {



    render () {

        console.log(this.props);

        return (
            <Modal trigger={
                            <Image size='medium' src={"data:image/png;base64," + this.props.results.explanation_image}/>
                            }>
                <Modal.Header>{this.props.results.model} trained on {this.props.results.dataset} dataset explained by
                    {this.props.results.interpreter}</Modal.Header>
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
                                    <Image size='small' src={"data:image/png;base64," + this.props.results.input_image} />
                                    <p>{this.props.results.ground_truth}</p>
                                </Table.Cell>
                                <Table.Cell>
                                    <Image size='small' src={"data:image/png;base64," + this.props.results.explanation_image}/>
                                    <p>{this.props.results.prediction}</p>
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    {this.props.results.explanation_text}
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    {this.props.results.start_time}
                                </Table.Cell>
                            </Table.Row>
                        </Table.Body>
                    </Table>
                </Modal.Content>
            </Modal>
        )
    }
}

export default ResultModal