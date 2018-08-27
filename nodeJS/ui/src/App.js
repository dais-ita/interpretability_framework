import React, { Component } from 'react';
import 'semantic-ui-css/semantic.min.css';
import './App.css';

import Title from "./components/Title";
import DatasetSelection from './components/DatasetSelection';
import ModelSelection from './components/ModelSelection';
import InterpretabilityComparison from "./components/InterpretabilityComparison";

class App extends Component {
    constructor(props) {
        super(props);
        this.state = {
            dataset: "",
            model: [],
            interpreter: []
        };
        this.setActiveDataset = this.setActiveDataset.bind(this);
        this.setActiveModel = this.setActiveModel.bind(this);
    }


    setActiveDataset(dataset) {
        this.setState({dataset:dataset})
    }

    setActiveModel(model) {
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

    }

    setActiveInterpreter(model) {

    }

    componentDidMount() {
        console.log("Current server: " + process.env.REACT_APP_SERVER_NAME + ":" + process.env.REACT_APP_PORT_NUMBER)
    }
    render() {
        return (
            <div className="App">
                <Title/>
                <DatasetSelection setActiveDataset={this.setActiveDataset} options={this.state}/>
                <ModelSelection setActiveModel={this.setActiveModel} options={this.state}/>
                <InterpretabilityComparison options={this.state}/>
            </div>
        );
  }
}

export default App;
