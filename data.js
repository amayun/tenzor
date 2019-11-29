import * as tf from '@tensorflow/tfjs';
import fetch from 'node-fetch';
import {PNG} from 'pngjs';
import {readFileSync, createWriteStream, existsSync, appendFileSync} from 'fs';

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGES_LOCAL_PATH = './images.png';
const LABELS_LOCAL_PATH = './labels.bin';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load() {
        // Make a request for the MNIST sprited image.
        const img = await this.loadImages();

        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * Float32Array.BYTES_PER_ELEMENT); // 65000 * 784 * 4
        this.datasetImages = new Float32Array(datasetBytesBuffer);
        const pixels = new Uint8ClampedArray(img.data);

        for (let j = 0; j < pixels.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            this.datasetImages[j] = pixels[j * 4] / 255;
        }

        const labels = await this.loadLabels();

        this.datasetLabels = new Uint8Array(labels);

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        // Slice the the images and labels into train and test sets.
        this.trainImages =
            this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.trainLabels =
            this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels =
            this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }

    async loadImages () {
        if(existsSync(IMAGES_LOCAL_PATH)) {
            const fileData = readFileSync(IMAGES_LOCAL_PATH);
            return  PNG.sync.read(fileData);
        } else {
            const imgRequest = await fetch(MNIST_IMAGES_SPRITE_PATH);
            const imgData = await imgRequest.arrayBuffer();
            const img = await new Promise((resolve, reject) => {
                new PNG().parse(imgData, (error, data) => {
                    if (error) {
                        reject(error)
                    }
                    resolve(data);
                });
            });

            img.pack().pipe(createWriteStream(IMAGES_LOCAL_PATH));

            return img;
        }
    }

    async loadLabels () {
        if(existsSync(LABELS_LOCAL_PATH)) {
            return readFileSync(LABELS_LOCAL_PATH);
        } else {
            const response = await fetch(MNIST_LABELS_PATH);
            const labels = await response.arrayBuffer();
            appendFileSync(LABELS_LOCAL_PATH, new Buffer(labels));

            return labels;
        }
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                this.shuffledTrainIndex =
                    (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex];
            });
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }

    nextBatch(batchSize, [imagesData, labelsData], index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image =
                imagesData.slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label =
                labelsData.slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]); // images by lines
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return {xs, labels};
    }
}