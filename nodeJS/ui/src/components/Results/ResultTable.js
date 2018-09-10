import React, { Component } from "react";
import {Grid, Image, Table} from "semantic-ui-react";
import ResultModal from "./ResultModal";
import _ from "lodash";
import "./results.css";

class ResultTable extends Component {

    state = {
        headings : []
    };

    componentDidMount() {
        console.log(this.props)
    }

    componentDidUpdate(prevProps) {

        if (prevProps !== this.props) {
            // console.log(this.props);
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

        //  _.times(this.props.results.length, i => (
        //     _.times(this.props.results[i].length, j => (
        //         const headings = this.state.headings;
        //         if (headings.indexOf(this.props.results[j].interpreter) !== -1) {
        //             headings.push(this.props.results[j].interpreter)
        //         }
        //         this.setState({headings})
        //     ))
        // ));

        // let input_image_preview;
        // if ((this.props.results.length > 0) && (typeof(this.props.results[0][0] !== "undefined"))) {
        //     input_image_preview = (
        //         <React.Fragment>
        //             <Image size='small' src={"data:image/png;base64," + this.props.results[0][0].input_image} />
        //             <p>{this.props.results[0][0].ground_truth}</p>
        //         </React.Fragment>
        //     )
        // } else {
        //     // input_image_preview
        // }

        // console.log(this.props.results);


        const interpreter_results = _.times(this.props.results.length, i => (
            <Table.Row key={i}>
                <Table.Cell>
                    {/*{console.log("### results: " + this.props.results[i][0])}*/}
                    {/*{input_image_preview}*/}
                    <Image size='medium' src={"data:image/png;base64," + this.props.inputs[i]} />
                    {/*<p>{this.props.results[i].ground_truth}</p>*/}
                </Table.Cell>
                {_.times(this.props.results[i].length, j => (
                    <Table.Cell key={j}>
                        <ResultModal results={this.props.results[i][j]}/>
                    </Table.Cell>
                ))
            }</Table.Row>
        ));


        return (
            <div id="results_display">
                <Table>
                    <Table.Body>
                    {/*<Table.Header>*/}
                        {/*<Table.Row>*/}
                            {/*<Table.HeaderCell>Input Image</Table.HeaderCell>*/}
                            {/*<Table.HeaderCell>{this.props.results[0][0].interpreter}</Table.HeaderCell>*/}
                            {/*<Table.HeaderCell>{this.props.results[0][1].interpreter}</Table.HeaderCell>*/}
                            {/*<Table.HeaderCell>{this.props.results[0][2].interpreter}</Table.HeaderCell>*/}
                            {/*<Table.HeaderCell>&nbsp;</Table.HeaderCell>*/}
                        {/*</Table.Row>*/}
                    {/*</Table.Header>*/}
                    {interpreter_results}
                    </Table.Body>
                </Table>

            </div>


        )



    }
}

export default ResultTable