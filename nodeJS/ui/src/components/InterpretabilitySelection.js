import React, { Component } from 'react';
import { Header, Image, Grid } from "semantic-ui-react";
import _ from 'lodash';


class InterpretabilitySelection extends Component {

    state = {

    };

    componentDidMount () {

    }


    render() {

        const columns = _.times(4, i => (
            <Grid.Column key={i}>
                <Image src='http://placehold.it/300x300' />
            </Grid.Column>
        ));

        return (
            <div>
                <Header as='h2'>3. Interpretability Technique Selection</Header>



            </div>
        );
    }
}

export default InterpretabilitySelection