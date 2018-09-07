import React, { Component } from "react";
import { Grid, Table } from "semantic-ui-react";
import ResultModal from "./ResultModal";
import _ from "lodash";
import "./results.css";

class ResultTable extends Component {

    componentDidMount() {
        console.log(this.props)
    }

    componentDidUpdate(prevProps) {

        if (prevProps !== this.props) {
            console.log(this.props);
            // alert("something updated")
            // console.log(this.props.results);
            // this.setState(this.state);
            //
            // _.times(this.props.results.length, i => (
            //     _.times(this.props.results[i].length, j => (
            //         console.log(this.props.results[i][j])
            //     ))
            // ))
        }
    }


    render () {

        //
        _.times(this.props.results.length, i => (
            _.times(this.props.results[i].length, j => (
                console.log(this.props.results[i][j])
            ))
        ));

        const interpreter_results = _.times(this.props.results.length, i => (
            <Table.Row key={i}>{
                _.times(this.props.results[i].length, j => (
                    <Table.Cell key={j}>
                        <ResultModal results={this.props.results[i][j]}/>
                    </Table.Cell>
                ))
            }</Table.Row>
        ));


        return (
            <div id="results_display">
                <Table>
                    <Table.Header>
                        <Table.Row>
                            <Table.HeaderCell>&nbsp;</Table.HeaderCell>
                            <Table.HeaderCell>&nbsp;</Table.HeaderCell>
                            <Table.HeaderCell>&nbsp;</Table.HeaderCell>
                            <Table.HeaderCell>&nbsp;</Table.HeaderCell>
                        </Table.Row>
                    </Table.Header>
                    {interpreter_results}
                </Table>

            </div>


        )



    }
}

export default ResultTable