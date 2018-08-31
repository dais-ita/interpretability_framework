import {Table, Button, Header} from "semantic-ui-react";
import React, { Component } from "react";

class InterpretabilityDescription extends Component  {
    constructor(props) {
        super(props);
    }

    state = {
        colour: "grey"
    };

    componentDidMount() {
        console.log("interpreter data");
    }

    componentDidUpdate(prevProps) {

        if (prevProps.options !== this.props.options) {
            if (this.props.options.interpreter.includes(this.props.interpreter_data.explanation_name)) {
                this.setState({colour: "green"})
            } else {
                this.setState({colour: "grey"})
            }
        }
    }


    render () {

        return (
            <Table.Row>

                <Table.Cell><Header as='h3'>{this.props.interpreter_data.explanation_name}</Header></Table.Cell>
                <Table.Cell>{this.props.interpreter_data.description}</Table.Cell>
                <Table.Cell>
                    <Button color={this.state.colour} onClick={ () => this.props.setActiveInterpreter(this.props.interpreter_data.explanation_name)}>
                        Use Interpreter
                    </Button>
                </Table.Cell>
            </Table.Row>

        //
        )

    }

}

export default InterpretabilityDescription



