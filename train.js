import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import {getModel, getSavedModel} from './model.js';
import {MnistData} from "./data.js";

import fs from 'fs';

const fitCallbacks = {
    onEpochEnd: (epoch, log) => {
        console.log('epoch end', log);
    }
};

let model;
let data;

export async function start() {
    console.log('start');

    await train();
    const res = await doPrediction();
    await printResult(res);
}

async function train() {
    data = new MnistData();
    await data.load();

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

    console.time('model training');
    const history = await model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
    console.timeEnd('model training');
    //model training: 1:47.268 (m:ss.mmm) (cpu)
    //model training: 1.294s  (cpu-node)

    await model.save('file://model');

    return history;
}

function doPrediction(testDataSize = 1000) {
    console.log('do prediction');

    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);

    const labels = testData.labels.argMax(1); // find index of activated class for each row
    const predictions = model.predict(testxs).argMax(1); // predict, then find most activated class for each row

    return {testxs, predictions, labels};
}

async function printResult({testxs, predictions, labels}) {
    const predsData = predictions.dataSync();
    const labelsData = labels.dataSync();
    const resPath = './res';

    try {
        fs.rmdirSync(resPath, {recursive: true});
        fs.mkdirSync(resPath);
    } catch {
    }

    let err = 0;
    const files = {};
    for (let i = 0; i < predsData.length; i++) {
        if (predsData[i] !== labelsData[i]) {
            err++;
            const imageTensor = tf.tidy(() => testxs.slice([i, 0], 1).reshape([28, 28]));
            const key = `pred-${predsData[i]}-label-${labelsData[i]}`;
            const finalKey = files[key] ? (files[key]++, `${key}_${files[key]}`) : (files[key] = 1, key);

            await data.printImageTensor(imageTensor, {writeToDisk: true}, `${resPath}/${finalKey}.png`);

            imageTensor.dispose();
        }
    }
    console.log(`total predictions ${predsData.length}, total errors ${err}`)
}
