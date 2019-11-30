import * as tf from '@tensorflow/tfjs-node';
import {getModel, getSavedModel, loadData} from './model';
import {printImage} from "./data";

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

    const err = 0;
    for (let i = 0; i < predsData.length; i++) {
        if (predsData[i] !== labelsData[i]) {
            const imageTensor = tf.tidy(() => testxs.slice([i, 0], [1, testxs.shape[1]]).reshape([28, 28, 1]));
            await printImage(imageTensor, 3, `res/pred-${predsData[i]}-label-${labelsData[i]}.png`);

            imageTensor.dispose();
        }
    }
}

console.log('start');

train()
    .then(doPrediction)
    .then(printResult);