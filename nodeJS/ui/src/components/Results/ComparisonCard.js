import {Header, Icon, Image, Segment, Table } from "semantic-ui-react";
import { } from "semantic-ui-react";

import React, { Component } from "react";

class ComparisonCard extends Component {
    state = {
      data: this.props.result_data
    };

    componentDidUpdate(prevProps) {
        // console.log(this.props);
        if (prevProps.options !== this.props.options) {
            this.setState({data: this.props.result_data})
        }
    }

    render () {

        console.log(this.props.result_data);

        const input_image = "data:image/png;base64," + this.state.data.input_image;
        const explain_image = "data:image/png;base64," + this.state.data.explanation_image;

        return (

            <div>
                <Segment padded compact>
                    <Header as='h3'>
                        {this.state.data.model} trained on {this.state.data.dataset} dataset explained by&nbsp;
                        {this.state.data.interpreter} &nbsp;<a><Icon name='down arrow'/></a> &nbsp;</Header>
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
                                    <Image size='small' src={input_image} />
                                </Table.Cell>
                                <Table.Cell>
                                    <Image size='small' src={explain_image}/>

                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    {this.state.data.explanation_text}
                                    {/*<p>Evidence towards predicted class shown in green</p>*/}
                                </Table.Cell>
                                <Table.Cell verticalAlign='top'>
                                    {this.state.data.start_time}
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
