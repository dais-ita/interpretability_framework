import React, { Component } from "react";
import { Grid } from "semantic-ui-react";
import ResultModal from "./ResultModal";
import _ from "lodash";
import "./results.css";

class ResultQueue extends Component {

    componentDidMount() {
        console.log(this.props.results.length)
    }

    componentDidUpdate(prevProps) {
        if (prevProps !== this.props) {
            // console.log(this.props.results)
        }
    }


    render (){


            const interpreter_results = _.times(this.props.results.length, i => (
                <Grid.Column key={i}>
                    <ResultModal results={this.props.results[i]}/>
                </Grid.Column>
            ));


        return (

            <Grid id="results_display" stackable columns={5}>
                {interpreter_results}
            </Grid>
        )



    }
}

export default ResultQueue