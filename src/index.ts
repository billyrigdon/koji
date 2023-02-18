import express from 'express';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
//import * as mm from 'music-metadata';
const mm = require("music-metadata")
import { normalizeData, sliceAudio } from 'model/koji';
import * as glob from "glob";
import sharp from 'sharp';

const app = express();
const DATA_DIR = "../../data";
const NUM_CLASSES = 24;
const SECONDS_PER_SLICE = 20;
const SAMPLE_RATE = 44100;

async function loadModel() {
  const model = await tf.loadLayersModel('file://model/model.json');
  return model;
}


async function predictKey(filename: string): Promise<number> {
	const file = glob.sync(`${DATA_DIR}/uploads/${filename}`)[0];
	const audioData = await fs.promises.readFile(file);
	const metadata = await mm.parseBuffer(audioData, { duration: true });
	const duration = metadata.format.duration as number;
	const numSlices = Math.floor(duration / SECONDS_PER_SLICE);
  const data: Float32Array[] = [];

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
	}
  const normalizedSlices = normalizeData(data);
  const model = await loadModel();
  const xs = tf.tensor3d(normalizedSlices.flat(2));
  const predictions = await model.predict(xs) as [];
  const avgPredictions = predictions.reduce((acc, curr) => {
    return acc.map((val, i) => val + curr[i]);
  }, new Array(NUM_CLASSES).fill(0)).map(val => val / predictions.length);
  const predictedKey = avgPredictions.indexOf(Math.max(...avgPredictions));
  return predictedKey;
}

app.post('/predict-key', async (req, res) => {
  try {
    const file = req.body.file;
    const buffer = await sharp(file.data).toFormat('png').toBuffer();
    const filepath = `./tmp/${file.name}`;
    fs.writeFileSync(filepath, buffer);
    const predictedKey = await predictKey(filepath);
    res.send(`Predicted key: ${predictedKey}`);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error predicting key');
  }
});

app.post('/train', async (req, res) => {
  // Get the audio file and label from the request body
  const { file, label } = req.body;

  // Slice the audio file into 20-second segments
  const audioData = await sliceAudio(file, 20);

  // Normalize the audio data
  const normalizedData = normalizeData(audioData);

  // Convert the normalized data to a tensor
  const tensorData = tf.tensor3d(normalizedData.flat(2));

  // Create a one-hot encoded label for the given key
  const labelTensor = tf.oneHot(tf.tensor1d([label]), 24);

  // Train the model on the preprocessed data and label
  const model = await loadModel();
  model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam' });
  await model.fit(tensorData, labelTensor);
  await model.save("file://model");
  // Send a response indicating that the model has been trained
  res.send('Model trained successfully');
});

app.listen(3000, () => {
  console.log('Server started on port 3000');
});
