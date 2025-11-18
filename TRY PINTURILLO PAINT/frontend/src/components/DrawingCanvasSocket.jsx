import { useRef, useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const DrawingCanvasSocket = ({ gameActive, currentWord, onCorrectGuess }) => {
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const socketRef = useRef(null);
  const videoImageRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [handLastPos, setHandLastPos] = useState({ x: 0, y: 0 });
  const [brushSize, setBrushSize] = useState(8);
  const [connected, setConnected] = useState(false);

  // Initialize socket connection
  useEffect(() => {
    const socket = io('http://localhost:5000', {
      transports: ['websocket', 'polling']
    });
    
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Connected to backend');
      setConnected(true);
      socket.emit('start_camera');
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setConnected(false);
    });

    socket.on('video_frame', (data) => {
      // Display video frame
      if (videoImageRef.current && data.frame) {
        videoImageRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }

      // Handle hand tracking data
      if (data.hand) {
        handleHandPosition(data.hand);
      }
    });

    socket.on('camera_started', (data) => {
      console.log('Camera started:', data);
    });

    return () => {
      socket.emit('stop_camera');
      socket.disconnect();
    };
  }, []);

  // Initialize canvases
  useEffect(() => {
    const canvas = canvasRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!canvas || !hiddenCanvas) return;

    const ctx = canvas.getContext('2d');
    const hiddenCtx = hiddenCanvas.getContext('2d');

    // Setup hidden canvas (white background for model)
    hiddenCtx.fillStyle = 'white';
    hiddenCtx.fillRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);
    hiddenCtx.strokeStyle = 'black';
    hiddenCtx.lineJoin = 'round';
    hiddenCtx.lineCap = 'round';
    hiddenCtx.lineWidth = brushSize;

    // Setup visible canvas
    ctx.strokeStyle = 'black';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = brushSize;
  }, [brushSize]);

  const handleHandPosition = (hand) => {
    if (!hand) return;

    const { x, y, isOpen } = hand;

    if (isOpen) {
      if (handLastPos.x !== 0 && handLastPos.y !== 0) {
        drawLine(handLastPos.x, handLastPos.y, x, y);
      }
      setHandLastPos({ x, y });
      schedulePrediction();
    } else {
      setHandLastPos({ x, y });
    }
  };

  const drawLine = (x1, y1, x2, y2) => {
    const canvas = canvasRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!canvas || !hiddenCanvas) return;

    const ctx = canvas.getContext('2d');
    const hiddenCtx = hiddenCanvas.getContext('2d');

    // Draw on visible canvas
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Draw on hidden canvas
    hiddenCtx.beginPath();
    hiddenCtx.moveTo(x1, y1);
    hiddenCtx.lineTo(x2, y2);
    hiddenCtx.stroke();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!canvas || !hiddenCanvas) return;

    const ctx = canvas.getContext('2d');
    const hiddenCtx = hiddenCanvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx.fillStyle = 'white';
    hiddenCtx.fillRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);
  };

  let predictionTimer = null;
  const schedulePrediction = () => {
    if (!gameActive || !currentWord) return;

    if (predictionTimer) clearTimeout(predictionTimer);
    predictionTimer = setTimeout(() => {
      predict();
    }, 1000);
  };

  const predict = async () => {
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!hiddenCanvas) return;

    // Check if canvas has any drawing
    const ctx = hiddenCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, hiddenCanvas.width, hiddenCanvas.height);
    const data = imageData.data;
    let isEmpty = true;
    
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] < 250 || data[i+1] < 250 || data[i+2] < 250) {
        isEmpty = false;
        break;
      }
    }
    
    if (isEmpty) return;

    const imageBase64 = hiddenCanvas.toDataURL('image/png');

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64 })
      });

      if (!response.ok) {
        console.error('Prediction failed:', response.statusText);
        return;
      }

      const result = await response.json();
      
      if (result.predictions && result.predictions.length > 0) {
        const topPrediction = result.predictions[0];
        if (topPrediction.class.toLowerCase() === currentWord.toLowerCase() && 
            topPrediction.percentage > 50) {
          onCorrectGuess();
        }
      }
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  // Mouse drawing handlers
  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    setIsDrawing(true);
    setLastPos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    drawLine(lastPos.x, lastPos.y, x, y);
    setLastPos({ x, y });
    schedulePrediction();
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ position: 'relative', width: '600px', height: '600px' }}>
        <img
          ref={videoImageRef}
          style={{
            position: 'absolute',
            width: '600px',
            height: '600px',
            objectFit: 'cover',
            borderRadius: '10px',
            zIndex: 1
          }}
          alt="Video feed"
        />
        <canvas
          ref={canvasRef}
          width={600}
          height={600}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{
            border: '3px solid #667eea',
            borderRadius: '10px',
            cursor: 'crosshair',
            display: 'block',
            position: 'relative',
            zIndex: 2
          }}
        />
      </div>
      <canvas
        ref={hiddenCanvasRef}
        width={600}
        height={600}
        style={{ display: 'none' }}
      />
      <div style={{ marginTop: '10px' }}>
        <div style={{ marginBottom: '10px', color: connected ? 'green' : 'red' }}>
          {connected ? '🟢 Connected' : '🔴 Disconnected'}
        </div>
        <label>
          Brush Size: 
          <input
            type="range"
            min="2"
            max="30"
            value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value))}
            style={{ marginLeft: '10px', marginRight: '10px' }}
          />
          {brushSize}px
        </label>
        <button onClick={clearCanvas} style={{ marginLeft: '20px' }}>
          Clear Canvas
        </button>
      </div>
    </div>
  );
};

export default DrawingCanvasSocket;
