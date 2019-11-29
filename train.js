import * as tf from '@tensorflow/tfjs-node';
import {getModel, loadData} from './model';

const fitCallbacks = {
    onEpochEnd: (epoch, log) => {
        console.log('log', log);
    }
};

let model;
let data;

async function train() {
    console.log('train start');

    model = getModel();
    data = await loadData();

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

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

function doPrediction(testDataSize = 500) {
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).argMax([-1]);

    return [testxs, preds, labels];
}

console.log('start');
train()
    .then(() => doPrediction())
    .then(([testxs, preds, labels]) => {
        const predsData = preds.dataSync();
        const labelsData = labels.dataSync();

        let res = '';
        for(let i = 0; i < predsData.length; i++) {
               res += `${predsData[i]} ${labelsData[i]}\n`
        }
        return res;

    })
    .then((res) => console.log(res));
