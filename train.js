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

    const history = await model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });

    await model.save('file://model');

    return history;
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

    let err = 0;
    for (let i = 0; i < predsData.length; i++) {
        if (predsData[i] !== labelsData[i]) {
            err++;
            const imageTensor = tf.tidy(() => testxs.slice([i, 0], [1, testxs.shape[1]]).reshape([28, 28, 1]));
            await printImage(imageTensor, 0, `res/pred-${predsData[i]}-label-${labelsData[i]}.png`);

            imageTensor.dispose();
        }
    }
    console.log(`total preds ${predsData.length}, total errors ${err}`)
}

console.log('start');

train()
    .then(() => doPrediction())
    .then((res) => printResult(res));