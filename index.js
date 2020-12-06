import {start} from './train.js';

await start();
//https://github.com/tensorflow/tfjs/issues/4116

/*

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
/!*const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);*!/

const d=[]; const s=[];
for (let i = 1; i<= 1000; i++) {
    if(i === 33) {
        continue;
    }

    d.push(i);
    s.push(i*i);
}

const xs = tf.tensor2d(d, [999, 1]);
const ys = tf.tensor2d(s, [999, 1]);

// Train the model using the data.
model.fit(xs, ys, {batchSize: 1, epochs: 2, verbose: 2}).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([33], [1, 1])).print();
});*/
