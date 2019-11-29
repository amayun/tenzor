import * as tf from '@tensorflow/tfjs-node';
import {getModel, getSavedModel, loadData} from './model';
import terminalImage from 'terminal-image';

const fitCallbacks = {
    onEpochEnd: (epoch, log) => {
        console.log('log', log);
    }
};

let model;
let data;

async function train() {
    data = await loadData();
    model = await getSavedModel();

    if (model) {
        return;
    }

    model = getModel();

    const BATCH_SIZE = 64;
    const trainDataSize = 500;
    const testDataSize = 100;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(trainDataSize);
        return [
            d.xs.reshape([trainDataSize, 28, 28, 1]),
            d.labels
        ]
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(testDataSize);
        return [
            d.xs.reshape([testDataSize, 28, 28, 1]),
            d.labels
        ]
    });

    model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
    model.save('file://model');
}

function doPrediction(testDataSize = 500) {
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).argMax([-1]);

    return [testxs, preds, labels];
}

async function printResult([testxs, preds, labels]) {
    const predsData = preds.dataSync();
    const labelsData = labels.dataSync();

    let res = '';
    for (let i = 0; i < predsData.length; i++) {
        res += `${predsData[i]} ${labelsData[i]}\n`
    }

    console.log(res);

    return res;
}

async function printImage() {
    console.log(await terminalImage.file('./demo.png'));
}

async function printX() {
    const testDataSize = 1;
    data = await loadData();

    const testData = data.nextTestBatch(testDataSize);
    //const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const testxs = testData.xs.reshape([testDataSize, 784, 1]);
    testxs.print();
    /*const xSData = testxs.dataSync();
    console.log('xSData', xSData);*/
}

console.log('start');
//printX();

train()
    .then(() => doPrediction())
    .then(printResult);

//printImage();