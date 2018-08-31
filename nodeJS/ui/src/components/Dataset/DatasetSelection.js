import React, { Component } from 'react';
import { Header, Image, Grid, Button, Transition, Icon } from "semantic-ui-react";
import _ from 'lodash';
import axios from 'axios';

import DatasetCard from "./DatasetCard";


class DatasetSelection extends Component {
    constructor(props) {
        super(props);
        this.state = {
            datasets: [],
            show_preview : false
        };
        this.togglePreview = this.togglePreview.bind(this);
    }


    componentDidMount () {
        const req = "http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER + "/datasets-all";
        axios.get(req)
            .then(res => {
                const datasets = res.data;
                this.setState( { datasets });
                this.props.setActiveDataset(datasets[0].dataset_name);
                // console.log(datasets[0].dataset_name)

                console.log(this.props.options);
            })
    }

    togglePreview = () => this.setState({show_preview: !this.state.show_preview});


    render() {

        const { show_preview } = this.state;

        const dataset_previews = _.times(this.state.datasets.length, i => (
            <Grid.Column key={i}>
               <DatasetCard setActiveDataset={this.props.setActiveDataset}
                            active_dataset={this.props.options.dataset}
                            dataset={this.state.datasets[i]}/>
            </Grid.Column>

        ));


        return (
            <div>
                <Header as='h2'>1. Dataset Selection</Header>
                <Grid container columns={4}>{dataset_previews}</Grid>
                {/*<div>*/}
                    {/*<Button content={show_preview ? 'Hide Preview' : 'Show Preview'} onClick={this.togglePreview}  />*/}
                    {/*<Button><Icon name='redo'/> </Button>*/}
                {/*</div>*/}

                {/*// todo*/}
                {/*<Grid columns={6}>*/}
                    {/*<Grid.Column>*/}
                        {/*<Transition visible={show_preview} animation='swing down' duration={200}>*/}
                            {/*<div>*/}
                                {/*<p>CODE FOR {this.props.options.dataset} PREVIEW GOES HERE</p>*/}
                                {/*<Image size='medium' src="http://placehold.it/600x600"/>*/}
                            {/*</div>*/}
                        {/*</Transition>*/}
                    {/*</Grid.Column>*/}


                {/*</Grid>*/}

            </div>
        );
    }
}

export default DatasetSelection