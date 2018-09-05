import {Button, Image, Segment, Form} from "semantic-ui-react";
import React, {Component} from "react";
import { Modal, Grid, Card, Divider, Header, Input } from "semantic-ui-react";
import axios from "axios";

import _ from "lodash";



class InputImageModal extends Component {
    state = { images: [], heroes: [], img: "", manual_img: "" };

    loadImages(dataset) {
        axios.get("http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
            "/" + "dataset-test-images?dataset=" + encodeURIComponent(dataset.trim())
            + "&num_images=100")
            .then(res => {
                this.setState({images: res.data})
            });

        axios.get("http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
            "/" + "dataset-details?dataset=" + encodeURIComponent(dataset.trim()))
            .then(res => {
                for (let i = 0; i < res.data.interesting_images.length; i++) {
                    axios.get("http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
                        "/" + "dataset-test-image?dataset=" + encodeURIComponent(dataset.trim()) +
                        "&image=" + res.data.interesting_images[i])
                        .then(res => {
                            let heroes = this.state.heroes;
                            heroes.push(res.data);
                            this.setState({heroes});
                            this.setState({manual_img: this.state.heroes[0]});
                            console.log(this.state.heroes);
                        })
                }

                console.log(res.data.interesting_images)
            });
    }

    componentDidUpdate() {
        console.log(this.props);
    }

    componentDidMount() {
        console.log(this.props);
    }

    handleSubmit = () => {
        axios.get("http://" + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER +
            "/" + "dataset-test-image?dataset=" + encodeURIComponent(this.props.dataset.trim()) +
            "&image=" + this.state.img)
            .then( res => {
                    this.setState({manual_img: res.data})
            });
    };

    handleChange = (e, {img, value}) => this.setState({ img: value });


    render() {

        const { img } = this.state;

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

        const hero_images = _.times(this.state.heroes.length, i => (
            <Grid.Column key={i}>
                <Card onClick={() => this.props.setInputImage(this.state.heroes[i])}>
                    <Image size='medium' src={"data:image/png;base64," + this.state.heroes[i].input}/>
                    <Card.Content>
                        <Card.Header>{this.state.heroes[i].image_name}</Card.Header>
                        <Card.Meta>{this.state.heroes[i].ground_truth}</Card.Meta>
                    </Card.Content>
                </Card>
            </Grid.Column>
        ));

        const manual_selection = (
            <div>
                <Form onSubmit={this.handleSubmit}>
                    <Form.Input placeholder='Image Name' name='img' onChange={this.handleChange} value={img}/>
                    <Form.Button type='submit'>Submit</Form.Button>
                </Form>

                <Card onClick={() => this.props.setInputImage(this.state.manual_img)}>
                    <Image size='medium' src={"data:image/png;base64," + this.state.manual_img.input}/>
                    <Card.Content>
                        <Card.Header>{this.state.manual_img.image_name}</Card.Header>
                        <Card.Meta>{this.state.manual_img.ground_truth}</Card.Meta>
                    </Card.Content>
                </Card>
            </div>
        );




        return (
            <div>
                <Modal closeIcon='close'
                       closeOnDocumentClick={true}
                       size='fullscreen'
                       onOpen={() => this.loadImages(this.props.dataset)}
                       trigger={

                           <Button>Choose Image</Button>

                       }>
                    <Modal.Header>{this.props.dataset}</Modal.Header>
                    <Modal.Content>
                        {manual_selection}
                        <Header>Interesting Images</Header>
                        <Grid stackable columns={4}>
                            {hero_images}
                        </Grid>

                        <Divider />
                        <Header>Random Images</Header>
                        <Grid stackable columns={4}>
                            {dataset_images}
                        </Grid>
                    </Modal.Content>

                </Modal>

            </div>

        );
    }
}

export default InputImageModal