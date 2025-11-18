# Quick Draw Challenge - Setup Instructions

## Backend Setup (Python)

1. Navigate to backend folder:
```bash
cd backend
```

2. Create virtual environment (if not exists):
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the backend:
```bash
python app.py
```

Backend will run on: http://localhost:5000

## Frontend Setup (React)

1. Open a new terminal and navigate to frontend folder:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the frontend:
```bash
npm run dev
```

Frontend will run on: http://localhost:5173

## What Changed - Backend Hand Tracking

### Improvements:
✅ **Much faster performance** - MediaPipe runs natively in Python (C++ backend)
✅ **More reliable** - No browser compatibility issues
✅ **Better FPS** - Optimized video processing
✅ **Lower latency** - Direct camera access
✅ **Less resource usage** - Browser doesn't need to process video

### Technology:
- **Backend**: Flask + SocketIO + MediaPipe + OpenCV
- **Frontend**: React + Socket.IO-client
- **Communication**: WebSockets for real-time video and hand tracking

### Features:
- Real-time hand tracking processed on server
- Video streaming with hand landmarks
- Open/closed hand detection
- Smooth drawing with palm center tracking
- Automatic game integration

Open http://localhost:5173 in your browser to play!
