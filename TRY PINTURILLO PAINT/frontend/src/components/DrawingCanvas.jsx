import { useRef, useEffect, useState, useCallback } from 'react';
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

const DrawingCanvas = ({ gameActive, currentWord, onCorrectGuess }) => {
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const videoRef = useRef(null);
  const handsRef = useRef(null);
  const cameraRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [handIsDrawing, setHandIsDrawing] = useState(false);
  const [handLastPos, setHandLastPos] = useState({ x: 0, y: 0 });
  const [brushSize, setBrushSize] = useState(8);
  const [shakeHistory, setShakeHistory] = useState([]);
  const [lastShakeTime, setLastShakeTime] = useState(0);

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

  const drawLine = useCallback((x1, y1, x2, y2) => {
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
  }, []);

  const checkHandOpen = useCallback((landmarks) => {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const distance = Math.sqrt(
      Math.pow(thumbTip.x - indexTip.x, 2) +
      Math.pow(thumbTip.y - indexTip.y, 2)
    );
    return distance > 0.1;
  }, []);

  const detectShake = useCallback((currentX) => {
    const now = Date.now();
    const newHistory = [...shakeHistory, { x: currentX, time: now }]
      .filter(pos => now - pos.time < 500);
    
    setShakeHistory(newHistory);

    if (newHistory.length < 10) return;

    let directionChanges = 0;
    let lastDirection = 0;

    for (let i = 1; i < newHistory.length; i++) {
      const diff = newHistory[i].x - newHistory[i - 1].x;
      if (Math.abs(diff) > 20) {
        const currentDirection = Math.sign(diff);
        if (lastDirection !== 0 && currentDirection !== lastDirection) {
          directionChanges++;
        }
        lastDirection = currentDirection;
      }
    }

    if (directionChanges >= 3 && now - lastShakeTime > 2000) {
      const canvas = canvasRef.current;
      const hiddenCanvas = hiddenCanvasRef.current;
      if (!canvas || !hiddenCanvas) return;

      const ctx = canvas.getContext('2d');
      const hiddenCtx = hiddenCanvas.getContext('2d');

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      hiddenCtx.fillStyle = 'white';
      hiddenCtx.fillRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);
      
      setLastShakeTime(now);
      setShakeHistory([]);
    }
  }, [shakeHistory, lastShakeTime]);

  const onHandResults = useCallback((results) => {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;

    const landmarks = results.multiHandLandmarks[0];
    const palmCenter = landmarks[9];

    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const x = (1 - palmCenter.x) * canvas.width;
    const y = palmCenter.y * canvas.height;

    detectShake(x);

    const isOpen = checkHandOpen(landmarks);

    if (isOpen) {
      setHandLastPos(prevPos => {
        // Draw line if we were already drawing
        if (handIsDrawing && prevPos.x !== 0 && prevPos.y !== 0) {
          drawLine(prevPos.x, prevPos.y, x, y);
        }
        return { x, y };
      });
      
      // Set drawing to true if not already
      setHandIsDrawing(prev => {
        if (!prev) {
          return true;
        }
        return prev;
      });
    } else {
      setHandIsDrawing(false);
    }
  }, [checkHandOpen, detectShake, drawLine, handIsDrawing]);

  // Initialize hand tracking
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    hands.onResults(onHandResults);
    handsRef.current = hands;

    const camera = new Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video });
      },
      width: 600,
      height: 600,
      frameRate: 60
    });

    camera.start();
    cameraRef.current = camera;

    return () => {
      camera.stop();
      hands.close();
    };
  }, [onHandResults]);

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

  const schedulePrediction = () => {
    if (!gameActive || !currentWord) return;

    setTimeout(() => {
      predict();
    }, 1000);
  };

  const predict = async () => {
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!hiddenCanvas) return;

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
      <video
        ref={videoRef}
        style={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          objectFit: 'cover',
          borderRadius: '10px',
          transform: 'scaleX(-1)',
          zIndex: 1
        }}
        autoPlay
        playsInline
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
      <canvas
        ref={hiddenCanvasRef}
        width={600}
        height={600}
        style={{ display: 'none' }}
      />
      <div style={{ marginTop: '10px' }}>
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

export default DrawingCanvas;
