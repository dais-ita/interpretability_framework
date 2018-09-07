import React, { Component } from 'react';
import { BrowserRouter as Router, Route } from "react-router-dom";
import {Accordion, Header, Icon, Button } from "semantic-ui-react";
import 'semantic-ui-css/semantic.min.css';
import './App.css';

import Title from "./components/Title";
import DatasetSelection from './components/Dataset/DatasetSelection';
import ModelSelection from './components/Model/ModelSelection';
import ExplainerSelection from "./components/Explainer/ExplainerSelection";
import ResultComparison from "./components/Results/ResultComparison";


class Home extends Component {
    constructor(props) {
        super(props);

        this.state = {
            explainer_comparison: false,
            use_case: 0,
            dataset: "",
            model: "",
            interpreter: "",
            activeIndex: 0
        };

        this.setActiveDataset = this.setActiveDataset.bind(this);
        this.setActiveModel = this.setActiveModel.bind(this);
        this.setActiveInterpreter = this.setActiveInterpreter.bind(this);
        this.toggleUseCase = this.toggleUseCase.bind(this);
    }

    componentDidMount() {
        console.log("Current server: " + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER)
    }

    setActiveDataset(dataset) {
        this.setState({dataset:dataset})
    }

    setActiveModel(model) {
        this.setState({model:model})
    }

    setActiveInterpreter(interpreter) {
        this.setState({interpreter:interpreter});
        console.log(this.state.interpreter);
    }

    toggleUseCase() {
        console.log(this.state);
        if (this.state.use_case === 0) {
            this.setState({use_case: 1 });
        } else {
            this.setState({use_case: 0 });
        }
    }

    handleClick = (e, titleProps) => {
        const { index } = titleProps;
        const { activeIndex } = this.state;
        const newIndex = activeIndex === index ? -1 : index;
        this.setState({ activeIndex: newIndex })
    };

    render() {
        const { activeIndex } = this.state;
        let case_toggle;
        let explainer_selection;

        if (this.state.use_case === 0) {
            case_toggle = (
                <div>
                    <Header as="h3">Use case: Explore Interpretability Techniques</Header>
                    <Button onClick={this.toggleUseCase}>Change mode</Button>
                </div>
            );

            explainer_selection = (
                <Accordion>
                    <Accordion.Title active={activeIndex === 2} index={2} onClick={this.handleClick}>
                        <Header as='h2'>
                            <Icon name='dropdown'/>
                            Interpretability Technique: &nbsp;
                            {this.state.interpreter}
                        </Header>
                    </Accordion.Title>
                    <Accordion.Content active={activeIndex === 2}>
                        <ExplainerSelection setActiveInterpreter={this.setActiveInterpreter} options={this.state}/>
                    </Accordion.Content>
                </Accordion>
            )
        } else {
            case_toggle = (
                <div>
                    <Header as="h3">Use case: Build Intuitions</Header>
                    <Button onClick={this.toggleUseCase}>Change mode</Button>
                </div>
            );

            explainer_selection = (
                <Accordion>
                    <Accordion.Title>
                        <Header as="h2">&nbsp;&nbsp;Interpretability Techniques: All</Header>
                    </Accordion.Title>
                </Accordion>
            )
        }

        return (
            <div className="App">
                <Title/>
                {case_toggle}
                <Accordion>
                        <Accordion.Title active={activeIndex === 0} index={0} onClick={this.handleClick}>
                            <Header as='h2'>
                                <Icon name='dropdown'/>
                                Dataset Selection: &nbsp;
                                {this.state.dataset}
                            </Header>
                        </Accordion.Title>
                        <Accordion.Content active={activeIndex === 0}>
                            <DatasetSelection setActiveDataset={this.setActiveDataset} options={this.state}/>
                        </Accordion.Content>
                </Accordion>

                <Accordion>
                    <Accordion.Title active={activeIndex === 1} index={1} onClick={this.handleClick}>
                        <Header as='h2'>
                            <Icon name='dropdown'/>
                            Model Selection: &nbsp;
                            {this.state.model}
                        </Header>
                    </Accordion.Title>
                    <Accordion.Content active={activeIndex === 1}>
                        <ModelSelection setActiveModel={this.setActiveModel} options={this.state}/>
                    </Accordion.Content>
                </Accordion>

                {explainer_selection}

                <ResultComparison options={this.state}/>

            </div>
        );
  }
}

const App = () => (
    <Router>
        <Route path="/(ui|)" component={Home} />
    </Router>
);

export default App;
