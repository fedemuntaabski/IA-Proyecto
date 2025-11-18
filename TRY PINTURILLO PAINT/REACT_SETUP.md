# Quick Draw Challenge - React + Vite Frontend

This is the new React-based frontend for the Quick Draw Challenge game.

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run the development server:
```bash
npm run dev
```

The frontend will run on http://localhost:5173

## Running the Full Application

You need both the backend (Flask) and frontend (React) running:

### Terminal 1 - Backend (Flask):
```bash
cd "d:\Tarea\4. INTELIGENCIA ARTIFICIAL\.TPs\TRY PINTURILLO PAINT"
python app.py
```

### Terminal 2 - Frontend (React):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Features

- ✅ React components with proper error handling
- ✅ Hand tracking with MediaPipe
- ✅ Mouse and touch drawing
- ✅ Game timer and scoring
- ✅ Auto-detection of drawings
- ✅ Shake gesture to clear
- ✅ Vite dev server with hot reload

## Technology Stack

- **React 19** - UI framework
- **Vite** - Build tool and dev server
- **MediaPipe Hands** - Hand tracking
- **Canvas API** - Drawing functionality
- **Flask** - Backend API (Python)
