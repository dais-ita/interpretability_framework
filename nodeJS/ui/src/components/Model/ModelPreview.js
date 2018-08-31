import {Table, Accordion, Image, Icon, Grid, Button} from "semantic-ui-react";
import React, { Component } from "react";

class ModelPreview extends Component  {
    constructor(props) {
        super(props);

    }

    state = {active: false};

    handleClick = (e, titleProps) => {
        const { active } = this.state;
        // const newIndex = activeIndex === index ? -1 : index;

        this.setState({ active: !this.state.active })
    };

    render () {
        const { active } = this.state;

        return (
            <Table.Row>
                <Table.Cell width='1'>
                </Table.Cell>
                <Table.Cell colSpan={2} >
                    <Accordion >
                        <Accordion.Title active={active} onClick={this.handleClick}>
                            <Icon name='dropdown' />
                            Performance Context
                        </Accordion.Title>
                        <Accordion.Content active={active}>
                            <Grid columns='2'>
                                <Grid.Column >
                                    <Image src='http://placehold.it/128x128' />
                                    <br/>
                                    Prediction: "Gun Wielder"
                                    <br/>
                                    Ground Truth: "Gun Wielder"

                                </Grid.Column>
                                <Grid.Column>
                                    <Button> New Image </Button>

                                </Grid.Column>
                            </Grid>
                        </Accordion.Content>
                    </Accordion>
                </Table.Cell>



            </Table.Row>

        )

    }

}

export default ModelPreview


