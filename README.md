# 🎙️ Speech Emotion Recognition System

A real-time Speech Emotion Recognition (SER) pipeline built with Python, Librosa, TensorFlow/Keras, and Matplotlib. The system captures live microphone audio, extracts acoustic spectral features, verifies active voice presence, and classifies the emotion behind the voice with visual probability plots.

---

## 📌 Features

- **Live Microphone Capture**: Record audio seamlessly using `sounddevice` and `soundfile`.
- **Acoustic Feature Extraction**: Extracts 67 audio features using `librosa`:
  - MFCCs (Mean & Standard Deviation)
  - Spectral Centroid & Spectral Rolloff
  - Zero Crossing Rate (ZCR)
  - RMS Energy
  - Chroma Features & Spectral Contrast
- **Voice Activity Detection (VAD)**: Filters out background silence and non-speech noise before inference.
- **Multi-Framework Model Loading**: Supports pre-trained `.h5`, `.keras` (TensorFlow/Keras), and `.pkl` (Scikit-Learn/Joblib) models.
- **Visual Analytics**: Interactive Matplotlib probability bar plots highlighting top predictions in real time.

---

## 📁 Directory Structure

```text
ml_notebook_emotion_voice_speech/
│
├── best_emotion_model.h5   # Pre-trained deep learning / ML model
├── feature_scaler.pkl       # Fitted StandardScaler / MinMaxScaler
├── emotion_recognizer.py    # Main Python application script
├── README.md                # Project documentation
└── recordings/              # Output folder for recorded WAV files (auto-created)

```

---

## 🛠️ Installation & Setup

### 1. Prerequisite Dependencies

Ensure you have **Python 3.8+** installed. Install all necessary dependencies via `pip`:

```bash
pip install numpy librosa sounddevice soundfile matplotlib joblib tensorflow

```

> **Note for PortAudio / Microphone input:**
> On Linux systems, you may need to install `portaudio`:
> ```bash
> sudo apt-get install python3-sounddevice libportaudio2
> 
> ```
> 
> 

---

## 🚀 How to Run

### Step 1: Ensure Model Assets Exist

Place your pre-trained model and feature scaler inside the project directory:

* `best_emotion_model.h5`
* `feature_scaler.pkl`

### Step 2: Run the Application

Execute the main script:

```bash
python emotion_recognizer.py

```

### Step 3: Speak into your Microphone

1. The terminal will display `Speak now...`.
2. Speak clearly into your microphone for the configured duration (default: 5 seconds).
3. The script will analyze the recording and output the emotion class probabilities both in the console and as a graphical popup.

---

## 📊 Supported Emotions

| Class ID | Label |
| --- | --- |
| `0` | Euphoric |
| `1` | Joyfully |
| `2` | Sad |
| `3` | Surprised |

---

## 💻 Example Output

### Console Output

```text
============================================================
MICROPHONE RECORDING
============================================================
Duration    : 5.0 seconds
Sample Rate : 22050

Speak now...
Recording completed successfully.

Checking for speech...

============================================================
ANALYZING AUDIO
============================================================
Estimated speech duration: 5.02 seconds

============================================================
EMOTION DETECTION RESULT
============================================================

Emotion    : sad
Confidence : 30.06%
Speech     : 5.02s

Emotion probabilities:
  sad             30.06%
  surprised       25.57%
  euphoric        22.53%
  joyfully        21.84%

============================================================

```

### Graphical Plot Output

An interactive `matplotlib` pop-up window will render a probability distribution chart highlighting the top predicted emotion.

---

## ⚙️ Configuration & Customization

You can adjust parameters inside `emotion_recognizer.py` when initializing `SpeechEmotionRecognizer`:

```python
recognizer = SpeechEmotionRecognizer(
    model_path="best_emotion_model.h5",  # Path to your .h5, .keras, or .pkl file
    scaler_path="feature_scaler.pkl",    # Path to feature scaler
    sample_rate=22050,                  # Target audio sample rate
    duration=5.0,                        # Recording duration in seconds
    delete_audio=False                   # Set to True to auto-delete audio clips after processing
)

```

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE&utm_source=gemini).

<ElicitationsGroup message="Next steps to complete your project setup:">
  <Elicitation label="Create a requirements.txt file" query="Generate a complete requirements.txt file for this Speech Emotion Recognition project."/>
  <Elicitation label="Add a training script for the model" query="Write a complete Python notebook/script to train this speech emotion model and export best_emotion_model.h5 and feature_scaler.pkl."/>
</ElicitationsGroup>

```
