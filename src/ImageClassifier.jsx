import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const ImageClassifier = () => {
  const [model, setModel] = useState(null);
  const [classificationResult, setClassificationResult] = useState('');

  const loadModel = async () => {
    // Load the converted TensorFlow.js model
    const loadedModel = await tf.loadLayersModel('../model/faceExpressionTFJS/model.json');
    setModel(loadedModel);
  };

  const handleImageUpload = async (event) => {
    setClassificationResult('') ;
    const file = event.target.files[0];
    const img = document.createElement('img');
    const reader = new FileReader();

    reader.onload = async () => {
      img.src = reader.result;
      img.onload = async () => {
        // Preprocess the image
        const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([256, 256]).toFloat().div(255.0).expandDims();

        // Make predictions
        const predictions = model.predict(tensor);
        const result = predictions.dataSync()[0];

        // Update UI with classification result
        console.log(result);
        setClassificationResult(result < 0.5 ? 'Happy' : 'Sad');
      };
    };

    reader.readAsDataURL(file);
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {model ? <p>Model Loaded</p> : <button onClick={loadModel}>Load Model</button>}
      {classificationResult && <p>Classification Result: {classificationResult}</p>}
    </div>
  );
};

export default ImageClassifier;
