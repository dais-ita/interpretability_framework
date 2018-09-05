import {Image, Modal, Table, Segment, Icon, Divider, Grid} from "semantic-ui-react";
import React, { Component } from "react";
import Moment from 'react-moment';
import moment from 'moment';


class ResultModal extends Component {

    componentDidMount () {

        let start = moment(this.props.results.start_time);
        console.log(start.diff(moment(this.props.results.end_time), "seconds"));

    }



    render () {

        console.log(this.props);

        let correct;

        if (this.props.results.ground_truth === this.props.results.prediction) {
            correct = <Icon color='green' name='thumbs up' />
        } else {
            correct = <Icon color='red' name='thumbs down' />
        }

        return (
            <Modal trigger={
                <div>
                            <Image size='medium' src={"data:image/png;base64," + this.props.results.explanation_image}/>
                    <p><strong>{this.props.results.interpreter}</strong> Prediction: {this.props.results.prediction}</p>
                </div>
                            }>
                <Modal.Header>
                    {this.props.results.model} trained on {this.props.results.dataset} dataset explained by&nbsp;
                    {this.props.results.interpreter}
                </Modal.Header>
                <Modal.Content>
                    {/*<Segment padded compact>*/}
                    {/*<Header as='h3'>{this.state.data.model} trained on {this.state.data.dataset} dataset explained by {this.state.data.interpreter} &nbsp;<a><Icon name='down arrow'/></a> &nbsp;</Header>*/}

                    Timestamp: <Moment format="DD/MM/YYYY HH:mm">{this.props.results.end_time}</Moment> &nbsp;&nbsp;&nbsp;

                    {"Duration: " + moment(this.props.results.start_time)
                        .diff(moment(this.props.results.end_time)) + "ms"}
                    <Divider/>

                    <Table basic='very'>
                        <Table.Header>
                            <Table.Row>
                                <Table.HeaderCell>Input Image</Table.HeaderCell>
                                <Table.HeaderCell>Explanation Image</Table.HeaderCell>
                                <Table.HeaderCell>Explanation Text</Table.HeaderCell>
                                <Table.HeaderCell>Prediction</Table.HeaderCell>
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
                                    <strong>Ground Truth:</strong> {this.props.results.ground_truth} <br/>
                                    <strong>Model Prediction:</strong> {this.props.results.prediction} &nbsp;  {correct}
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