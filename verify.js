const fs = require('fs');

class NeuralNetwork {
    constructor(Wih, Who) {
        this.inputNodes = 784;
        this.hiddenNodes = 200;
        this.outputNodes = 10;

        let flatWih = [];
        for (let i = 0; i < Wih.length; i++) {
            for (let j = 0; j < Wih[i].length; j++) {
                flatWih.push(Wih[i][j]);
            }
        }
        this.Wih = new Float32Array(flatWih);

        let flatWho = [];
        for (let i = 0; i < Who.length; i++) {
            for (let j = 0; j < Who[i].length; j++) {
                flatWho.push(Who[i][j]);
            }
        }
        this.Who = new Float32Array(flatWho);
    }

    sigmoid(x) {
        if (x < -500) return 0;
        if (x > 500) return 1;
        return 1 / (1 + Math.exp(-x));
    }

    forward(inputs) {
        const hiddenOutputs = new Float32Array(this.hiddenNodes);
        for (let j = 0; j < this.hiddenNodes; j++) {
            let sum = 0;
            for (let i = 0; i < this.inputNodes; i++) {
                sum += inputs[i] * this.Wih[i * this.hiddenNodes + j];
            }
            hiddenOutputs[j] = this.sigmoid(sum);
        }

        const finalOutputs = new Float32Array(this.outputNodes);
        for (let j = 0; j < this.outputNodes; j++) {
            let sum = 0;
            for (let i = 0; i < this.hiddenNodes; i++) {
                sum += hiddenOutputs[i] * this.Who[i * this.outputNodes + j];
            }
            finalOutputs[j] = this.sigmoid(sum);
        }

        return finalOutputs;
    }
}

const w = JSON.parse(fs.readFileSync('weights.json', 'utf8'));
const nn = new NeuralNetwork(w.Wih, w.Who);

const csv = fs.readFileSync('test-10.csv', 'utf8').trim().split('\n');
for (let line of csv) {
    let parts = line.split(',');
    let label = parseInt(parts[0]);
    let inputs = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
        inputs[i] = (parseFloat(parts[i + 1]) / 255.0 * 0.99) + 0.01;
    }

    let outputs = nn.forward(inputs);
    let maxIdx = 0;
    let maxVal = outputs[0];
    for (let i = 1; i < 10; i++) {
        if (outputs[i] > maxVal) {
            maxVal = outputs[i];
            maxIdx = i;
        }
    }
    console.log(`True: ${label}, Pred: ${maxIdx}`);
}
