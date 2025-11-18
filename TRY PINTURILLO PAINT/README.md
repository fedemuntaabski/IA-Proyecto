# Sketch Classifier App

A simple web application that lets you draw sketches and uses a Keras model to guess what you're drawing in real-time.

## Features

- 🎨 Interactive drawing canvas
- 🤖 Real-time sketch classification
- 📊 Top 5 predictions with confidence scores
- 🖱️ Mouse and touch support
- 🎯 Adjustable brush size

## Setup

1. A virtual environment has been created for this project. All dependencies are installed in `venv/`.

2. To run the app, simply use the provided batch file:
```bash
run.bat
```

   Or manually activate the virtual environment and run:
   ```bash
   venv\Scripts\activate
   python app.py
   ```

3. Open your browser and go to:
```
http://localhost:5000
```

### Manual Setup (if needed)

If you need to recreate the virtual environment:
```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

1. Draw anything on the canvas
2. The app will automatically predict what you're drawing after you finish
3. See the top 5 predictions with confidence percentages
4. Use "Clear Canvas" to start over
5. Use "Predict Now" to force an immediate prediction

## Notes

- The model expects 28x28 grayscale images (standard for Quick Draw dataset)
- You may need to adjust the `CLASS_NAMES` list in `app.py` to match your model's training classes
- The app automatically predicts 1 second after you stop drawing

## Files

- `app.py` - Flask backend server
- `index.html` - Main HTML interface
- `app.js` - Drawing and prediction logic
- `sketch_classifier_model.keras` - Your trained Keras model
- `requirements.txt` - Python dependencies
