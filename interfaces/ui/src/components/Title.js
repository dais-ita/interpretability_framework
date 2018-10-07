import React, { Component } from "react";
import { Header } from "semantic-ui-react";


class Title extends Component {
    render() {
        return (
            <div>
                <Header as="h1">Interpretable Machine Learning Zoo</Header>
                <p>
                    {/*<strong>"Interpretability is the degree to which a human can understand the cause of a decision" - Miller, 2017<br/></strong>*/}
                </p>
            </div>
        );
    }
}

export default Title