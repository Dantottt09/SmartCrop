
import React, { useState } from 'react';

import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Alert,
  StatusBar,
  ActivityIndicator
} from 'react-native';

import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';

import { useTensorflowModel } from 'react-native-fast-tflite';
import { Buffer } from 'buffer';

const jpeg = require('jpeg-js');

const MODEL_SIZE = 224;

const labels = require('./assets/model/class_indices.json');

export default function App() {

  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  const modelPlugin = useTensorflowModel(
    require('./assets/model/1.tflite'),
    []
  );

  const model =
    modelPlugin.state === 'loaded'
      ? modelPlugin.model
      : null;

  const openCamera = async () => {

    const permission =
      await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert(
        'Permission Required',
        'Please allow camera access.'
      );
      return;
    }

    const result =
      await ImagePicker.launchCameraAsync({
        mediaTypes: ['images'],
        quality: 1
      });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setPrediction(null);
      setConfidence(null);
    }
  };

  const openGallery = async () => {

    const permission =
      await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert(
        'Permission Required',
        'Please allow gallery access.'
      );
      return;
    }

    const result =
      await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 1
      });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setPrediction(null);
      setConfidence(null);
    }
  };

  // Convert image into a 224 x 224 RGB tensor.
  const prepareImage = async (uri) => {

    const resized =
      await ImageManipulator.manipulateAsync(
        uri,
        [
          {
            resize: {
              width: MODEL_SIZE,
              height: MODEL_SIZE
            }
          }
        ],
        {
          compress: 1,
          format: ImageManipulator.SaveFormat.JPEG,
          base64: true
        }
      );

    if (!resized.base64) {
      throw new Error('Unable to process image.');
    }

    const decoded = jpeg.decode(
      Buffer.from(resized.base64, 'base64'),
      { useTArray: true }
    );

    const pixels = decoded.data;

    const tensor = new Int32Array(
      MODEL_SIZE * MODEL_SIZE * 3
    );

    let index = 0;

    for (let i = 0; i < pixels.length; i += 4) {

      tensor[index++] = pixels[i];
      tensor[index++] = pixels[i + 1];
      tensor[index++] = pixels[i + 2];

    }

    return tensor.buffer;
  };

  // Run TensorFlow Lite disease prediction.
  const analyzePlant = async () => {

    if (!image) {
      Alert.alert(
        'No Image',
        'Please capture or select a plant image first.'
      );
      return;
    }

    if (!model) {
      Alert.alert(
        'Model Loading',
        'Please wait until the AI model is ready.'
      );
      return;
    }

    try {

      setLoading(true);
      setPrediction(null);
      setConfidence(null);

      const inputBuffer = await prepareImage(image);

      const outputs = await model.run([
        inputBuffer
      ]);

      const scores = new Float32Array(outputs[0]);

      if (scores.length !== 38) {
        throw new Error(
          'Unexpected model output. Expected 38 classes, received ' +
          scores.length
        );
      }

      let probabilities = Array.from(scores);

      // Apply softmax if model output is not already probabilities.
      const sum = probabilities.reduce(
        (total, value) => total + value,
        0
      );

      const isProbabilityOutput =
        probabilities.every(
          value => value >= 0 && value <= 1
        ) &&
        Math.abs(sum - 1) < 0.05;

      if (!isProbabilityOutput) {

        const maxScore = Math.max(...probabilities);

        const exponentials = probabilities.map(
          value => Math.exp(value - maxScore)
        );

        const total = exponentials.reduce(
          (a, b) => a + b,
          0
        );

        probabilities = exponentials.map(
          value => value / total
        );
      }

      let bestIndex = 0;

      for (let i = 1; i < probabilities.length; i++) {

        if (
          probabilities[i] >
          probabilities[bestIndex]
        ) {
          bestIndex = i;
        }

      }

      const detectedClass =
        labels[String(bestIndex)];

      if (!detectedClass) {
        throw new Error(
          'Disease label not found in class_indices.json'
        );
      }

      setPrediction(
        detectedClass.replace(/___/g, ' - ').replace(/_/g, ' ')
      );

      setConfidence(
        (probabilities[bestIndex] * 100).toFixed(2)
      );

    } catch (error) {

      console.log('Detection Error:', error);

      Alert.alert(
        'Detection Failed',
        error.message || 'Unable to analyze the plant image.'
      );

    } finally {

      setLoading(false);

    }
  };

  const removeImage = () => {

    setImage(null);
    setPrediction(null);
    setConfidence(null);

  };

  return (

    <View style={styles.container}>

      <StatusBar
        backgroundColor="#F5FAF3"
        barStyle="dark-content"
      />

      <ScrollView
        showsVerticalScrollIndicator={false}
      >

        {/* HEADER */}

        <View style={styles.header}>

          <View>
            <Text style={styles.logo}>
              🌱 SmartCrop
            </Text>

            <Text style={styles.subtitle}>
              Your Plant Health Assistant
            </Text>
          </View>

          <Text style={styles.leaf}>
            🍃
          </Text>

        </View>

        {/* WELCOME */}

        <View style={styles.welcome}>

          <Text style={styles.welcomeTitle}>
            Keep Your Plants Healthy!
          </Text>

          <Text style={styles.welcomeText}>
            Identify plant diseases by taking a photo
            of your plant leaves.
          </Text>

        </View>

        {/* SCANNER */}

        <View style={styles.card}>

          <Text style={styles.cardTitle}>
            Plant Scanner
          </Text>

          <Text style={styles.cardSubtitle}>
            Capture or upload a plant image
          </Text>

          {image ? (

            <Image
              source={{ uri: image }}
              style={styles.preview}
            />

          ) : (

            <View style={styles.placeholder}>

              <Text style={styles.cameraIcon}>
                📷
              </Text>

              <Text style={styles.placeholderText}>
                Your plant image will appear here
              </Text>

            </View>

          )}

          {/* CAMERA */}

          <TouchableOpacity
            style={styles.primaryButton}
            onPress={openCamera}
          >

            <Text style={styles.buttonText}>
              📷  Take a Photo
            </Text>

          </TouchableOpacity>

          {/* GALLERY */}

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={openGallery}
          >

            <Text style={styles.secondaryText}>
              🖼️  Choose from Gallery
            </Text>

          </TouchableOpacity>

          {/* ANALYZE */}

          {image && (

            <>

              <TouchableOpacity
                style={styles.analyzeButton}
                onPress={analyzePlant}
                disabled={loading || !model}
              >

                {loading ? (

                  <ActivityIndicator color="#FFFFFF" />

                ) : (

                  <Text style={styles.buttonText}>
                    🔍  Analyze Plant
                  </Text>

                )}

              </TouchableOpacity>

              <TouchableOpacity
                style={styles.removeButton}
                onPress={removeImage}
              >

                <Text style={styles.removeText}>
                  Remove Image
                </Text>

              </TouchableOpacity>

            </>

          )}

          {/* MODEL STATUS */}

          <View style={styles.modelStatus}>

            {modelPlugin.state === 'loading' ? (

              <Text style={styles.statusText}>
                ⏳ Loading AI model...
              </Text>

            ) : modelPlugin.state === 'loaded' ? (

              <Text style={styles.readyText}>
                ✓ AI Model Ready
              </Text>

            ) : modelPlugin.state === 'error' ? (

              <Text style={styles.errorText}>
                Model Error: {modelPlugin.error?.message}
              </Text>

            ) : null}

          </View>

        </View>

        {/* RESULTS */}

        {prediction && (

          <View style={styles.resultBox}>

            <Text style={styles.resultTitle}>
              🌿 Disease Detection Result
            </Text>

            <Text style={styles.diseaseName}>
              {prediction}
            </Text>

            <Text style={styles.confidence}>
              Confidence: {confidence}%
            </Text>

            <Text style={styles.resultNote}>
              This result is based on the prediction
              of the TensorFlow Lite model.
              Verify the result before applying treatment.
            </Text>

          </View>

        )}

        {/* INFORMATION */}

        <View style={styles.infoCard}>

          <Text style={styles.infoTitle}>
            Why SmartCrop?
          </Text>

          <Text style={styles.infoText}>
            🌿  Easy plant scanning
          </Text>

          <Text style={styles.infoText}>
            📱  Mobile-friendly application
          </Text>

          <Text style={styles.infoText}>
            ⚡  Designed for offline detection
          </Text>

          <Text style={styles.infoText}>
            🤖  AI-powered plant disease classification
          </Text>

        </View>

        {/* FOOTER */}

        <Text style={styles.footer}>
          SmartCrop © 2026 | Plant Health Made Smarter
        </Text>

      </ScrollView>

    </View>

  );

}

const styles = StyleSheet.create({

  container: {
    flex: 1,
    backgroundColor: '#F5FAF3'
  },

  header: {
    padding: 24,
    paddingTop: 35,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center'
  },

  logo: {
    fontSize: 27,
    fontWeight: 'bold',
    color: '#236B3B'
  },

  subtitle: {
    color: '#718575',
    marginTop: 4,
    fontSize: 13
  },

  leaf: {
    fontSize: 35
  },

  welcome: {
    backgroundColor: '#236B3B',
    margin: 18,
    marginTop: 5,
    padding: 24,
    borderRadius: 20
  },

  welcomeTitle: {
    fontSize: 22,
    color: '#FFFFFF',
    fontWeight: 'bold'
  },

  welcomeText: {
    color: '#E5F3E7',
    marginTop: 10,
    lineHeight: 22
  },

  card: {
    backgroundColor: '#FFFFFF',
    margin: 18,
    marginTop: 5,
    padding: 20,
    borderRadius: 18,
    elevation: 3
  },

  cardTitle: {
    fontSize: 21,
    fontWeight: 'bold',
    color: '#245B35'
  },

  cardSubtitle: {
    color: '#829084',
    marginTop: 5,
    marginBottom: 18
  },

  placeholder: {
    height: 190,
    backgroundColor: '#EDF5ED',
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#D4E5D5',
    borderStyle: 'dashed'
  },

  cameraIcon: {
    fontSize: 45
  },

  placeholderText: {
    color: '#829084',
    marginTop: 10,
    fontSize: 13
  },

  preview: {
    width: '100%',
    height: 220,
    borderRadius: 15,
    resizeMode: 'cover'
  },

  primaryButton: {
    backgroundColor: '#287A42',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 18
  },

  buttonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 16
  },

  secondaryButton: {
    borderWidth: 1,
    borderColor: '#287A42',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 12
  },

  secondaryText: {
    color: '#287A42',
    fontWeight: 'bold',
    fontSize: 15
  },

  analyzeButton: {
    backgroundColor: '#164D2A',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 15
  },

  removeButton: {
    marginTop: 14,
    alignSelf: 'center'
  },

  removeText: {
    color: '#B04444',
    fontWeight: 'bold'
  },

  modelStatus: {
    marginTop: 15,
    alignItems: 'center'
  },

  statusText: {
    color: '#829084',
    fontSize: 12
  },

  readyText: {
    color: '#287A42',
    fontWeight: 'bold',
    fontSize: 13
  },

  errorText: {
    color: '#B04444',
    fontSize: 12
  },

  resultBox: {
    backgroundColor: '#EDF7ED',
    margin: 18,
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#C9E2CC'
  },

  resultTitle: {
    color: '#236B3B',
    fontWeight: 'bold',
    fontSize: 18
  },

  diseaseName: {
    color: '#245B35',
    fontSize: 19,
    fontWeight: 'bold',
    marginTop: 15
  },

  confidence: {
    color: '#287A42',
    fontWeight: 'bold',
    marginTop: 10
  },

  resultNote: {
    color: '#53675A',
    marginTop: 12,
    lineHeight: 21
  },

  infoCard: {
    margin: 18,
    marginTop: 0,
    padding: 20,
    backgroundColor: '#E5F1E5',
    borderRadius: 16
  },

  infoTitle: {
    fontSize: 19,
    fontWeight: 'bold',
    color: '#245B35',
    marginBottom: 12
  },

  infoText: {
    color: '#47634D',
    marginVertical: 5,
    fontSize: 14
  },

  footer: {
    textAlign: 'center',
    color: '#89978B',
    fontSize: 12,
    margin: 25
  }

});
