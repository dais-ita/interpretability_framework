import React, { Component } from 'react';
import { BrowserRouter as Router, Route } from "react-router-dom";
import {Accordion, Header, Icon, Grid} from "semantic-ui-react";
import 'semantic-ui-css/semantic.min.css';
import './App.css';

import Title from "./components/Title";
import DatasetSelection from './components/Dataset/DatasetSelection';
import ModelSelection from './components/Model/ModelSelection';
import InterpretabilitySelection from "./components/Interpreter/InterpretabilitySelection";
import ResultComparison from "./components/Results/ResultComparison";


class Home extends Component {
    constructor(props) {
        super(props);

        this.state = {
            comparison_mode: false,
            dataset: "",
            model: "",
            interpreter: "",
            activeIndex: 0
        };

        this.setActiveDataset = this.setActiveDataset.bind(this);
        this.setActiveModel = this.setActiveModel.bind(this);
        this.setActiveInterpreter = this.setActiveInterpreter.bind(this);
    }

    componentDidMount() {
        console.log("Current server: " + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER)
    }

    setActiveDataset(dataset) {
        this.setState({dataset:dataset})
    }

    setActiveModel(model) {

        if (this.state.comparison_mode) {
            const model_list = this.state.model;

            if (model_list.includes(model)) {
                const model_index = model_list.indexOf(model);
                model_list.splice(model_index, 1);
                this.setState({model: model_list});
            } else {
                model_list.push(model);
                this.setState({model:model_list});
            }
            console.log(this.state.model);
        } else {
            this.setState({model:model})
        }

    }

    setActiveInterpreter(interpreter) {

        if (this.state.comparison_mode) {
            const interpreter_list = this.state.interpreter;

            if (interpreter_list.includes(interpreter)) {
                const interpreter_index = interpreter_list.indexOf(interpreter);
                interpreter_list.splice(interpreter_index, 1);
                this.setState({interpreter: interpreter_list});
            } else {
                interpreter_list.push(interpreter);
                this.setState({interpreter: interpreter_list});
            }

            console.log(this.state.interpreter);
        } else {
            this.setState({interpreter:interpreter});
        }
    }

    handleClick = (e, titleProps) => {
        const { index } = titleProps
        const { activeIndex } = this.state
        const newIndex = activeIndex === index ? -1 : index

        this.setState({ activeIndex: newIndex })
    };

    render() {
        const { activeIndex } = this.state;

        return (
            <div className="App">
                <Title/>
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

                <Accordion>
                    <Accordion.Title active={activeIndex === 2} index={2} onClick={this.handleClick}>
                        <Header as='h2'>
                            <Icon name='dropdown'/>
                            Interpretability Technique: &nbsp;
                            {this.state.interpreter}
                        </Header>
                    </Accordion.Title>
                    <Accordion.Content active={activeIndex === 2}>
                        <InterpretabilitySelection setActiveInterpreter={this.setActiveInterpreter} options={this.state}/>
                    </Accordion.Content>
                </Accordion>




                <ResultComparison options={this.state}/>

            </div>
        );
  }
}

const App = () => (
    <Router>
        <Route path="/ui" component={Home} />
        {/*<Route path="/" component={Home} />*/}
    </Router>
);

export default App;
