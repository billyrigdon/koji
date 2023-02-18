import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";
//import * as mm from "music-metadata";
const mm = require("music-metadata")
import * as glob from "glob";

const DATA_DIR = "../../data";
const NUM_CLASSES = 24;
const SECONDS_PER_SLICE = 20;
const SAMPLE_RATE = 44100;

// Helper function to slice audio data into 20-second chunks
export const sliceAudio = (
	audioData: Float32Array,
	sampleRate: number
): Float32Array[] => {
	const sliceSize = sampleRate * SECONDS_PER_SLICE;
	const numSlices = Math.floor(audioData.length / sliceSize);
	const slices: Float32Array[] = [];

	for (let i = 0; i < numSlices; i++) {
		const start = i * sliceSize;
		const end = (i + 1) * sliceSize;
		const slice = audioData.slice(start, end);
		slices.push(slice);
	}

	return slices;
};

// Load all MP3 files in the data directory and extract features
export const extractFeatures = async (): Promise<{
	data: Float32Array[];
	labels: number[];
}> => {
	const files = glob.sync(`${DATA_DIR}/**/*.mp3`);
	const data: Float32Array[] = [];
	const labels: number[] = [];

	for (let i = 0; i < files.length; i++) {
		const file = files[i];
		const key = path.basename(path.dirname(file));
		const audioData = await fs.promises.readFile(file);
		const metadata = await mm.parseBuffer(audioData, { duration: true });
		const duration = metadata.format.duration as number;
		const numSlices = Math.floor(duration / SECONDS_PER_SLICE);

		for (let j = 0; j < numSlices; j++) {
			const start = j * SECONDS_PER_SLICE;
			const end = (j + 1) * SECONDS_PER_SLICE;
			const sliceData = audioData.slice(
				start * SAMPLE_RATE * 2,
				end * SAMPLE_RATE * 2
			);
			const slice = sliceAudio(
				new Float32Array(sliceData.buffer),
				SAMPLE_RATE
			);
			data.push(...slice);
			labels.push(parseInt(key));
		}
	}

	return { data, labels };
};

// Normalize the data by subtracting the mean and dividing by the standard deviation
export const normalizeData = (data: Float32Array[]): number[][] => {
	const mean = tf.mean(data).dataSync()[0];
	const std = Math.sqrt(tf.moments(data).variance.dataSync()[0]);
	return data.map((row) => Array.from(row).map((val) => (val - mean) / std));
}

// Convert labels to one-hot encoding
const toOneHot = (labels: number[]): number[][] => {
	const oneHot: number[][] = [];
	for (let i = 0; i < labels.length; i++) {
		const arr = new Array(NUM_CLASSES).fill(0);
		arr[labels[i]] = 1;
		oneHot.push(arr);
	}
	return oneHot;
};

function trainModel(data: number[][], labels: number[]) {
	const flattenedData = data.flat(2);
	const xs = tf.tensor3d(flattenedData);
	const ys = tf.oneHot(tf.tensor1d(labels, "int32"), NUM_CLASSES);

	const model = tf.sequential();
	model.add(tf.layers.flatten({ inputShape: [xs.shape[1], xs.shape[2]] }));
	model.add(tf.layers.dense({ units: 64, activation: "relu" }));
	model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));
	model.compile({
		optimizer: "adam",
		loss: "categoricalCrossentropy",
		metrics: ["accuracy"],
	});
	model.fit(xs, ys, { epochs: 10 }).then(() => {
		model.save("file://model");
	});
}

const main = async () => {
	const { data, labels } = await extractFeatures();
	const normalizedData = normalizeData(data);
	await trainModel(normalizedData, labels);
};

main();
