import {Button, Image, Segment} from "semantic-ui-react";
import React, {Component} from "react";
import { Modal, Grid, Card } from "semantic-ui-react";
import axios from "axios";

import _ from "lodash";


class ChooseInputImage extends Component {
    state = { images: [] };

    loadImages(dataset) {
        axios.get("http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
            "/" + "dataset-test-images?dataset=" + encodeURIComponent(dataset.trim())
            + "&num_images=100")
            .then(res => {
                this.setState({images: res.data})
            });
    }

    componentDidUpdate() {
        console.log(this.props);
    }

    componentDidMount() {
        console.log(this.props);
    }

    render() {
        const dataset_images = _.times(this.state.images.length, i => (
            <Grid.Column key={i}>
                <Card onClick={() => this.props.setInputImage(this.state.images[i])}>
                    <Image size='medium' src={"data:image/png;base64," + this.state.images[i].input}/>
                    <Card.Content>
                        <Card.Header>{this.state.images[i].image_name}</Card.Header>
                        <Card.Meta>{this.state.images[i].ground_truth}</Card.Meta>
                    </Card.Content>
                </Card>
            </Grid.Column>
        ));






        return (
            <div>
                <Modal closeIcon='close'
                       closeOnDocumentClick={true}
                       centered={true}
                       size='fullscreen'
                       onOpen={() => this.loadImages(this.props.dataset)}
                       trigger={<Button>Choose Image</Button>}>
                    <Modal.Header>{this.props.dataset}</Modal.Header>
                    <Modal.Content>
                        <Grid stackable columns={5}>
                            {dataset_images}
                        </Grid>
                    </Modal.Content>

                </Modal>

            </div>

        );
    }
}

export default ChooseInputImage