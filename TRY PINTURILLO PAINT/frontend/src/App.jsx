import { useState, useEffect } from 'react'
import DrawingCanvasSocket from './components/DrawingCanvasSocket'
import GameStatus from './components/GameStatus'
import './App.css'

const GAME_CLASSES = [
  "The Eiffel Tower", "The Great Wall of China", "The Mona Lisa", "aircraft carrier", "airplane",
  "alarm clock", "ambulance", "arm", "asparagus", "axe", "baseball", "baseball bat", "basket",
  "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed", "bee", "belt", "bench",
  "bicycle", "binoculars", "bird", "birthday cake", "book", "boomerang", "bowtie", "bracelet",
  "brain", "bread", "bridge", "broccoli", "bucket", "bulldozer", "bus", "bush", "butterfly",
  "cactus", "cake", "camel", "camera", "cannon", "canoe", "carrot", "cat", "ceiling fan",
  "cell phone", "chair", "chandelier", "church", "computer", "cookie", "couch", "crab", "crown",
  "dolphin", "donut", "drill", "drums", "duck", "dumbbell", "ear", "elephant", "envelope",
  "eraser", "fan", "feather", "fence", "finger", "firetruck", "fish", "flamingo", "flashlight",
  "flip flops", "floor lamp", "flower", "foot", "fork", "frog", "frying pan", "garden",
  "giraffe", "golf club", "grapes", "grass", "guitar", "hamburger", "hammer", "hand", "harp",
  "hat", "headphones", "helicopter", "helmet", "hexagon", "hot air balloon", "hot dog", "house",
  "jacket", "jail", "knee", "knife", "ladder", "lantern", "laptop", "leaf", "leg", "light bulb",
  "lightning", "lion", "lipstick", "lobster", "lollipop", "map", "matches", "megaphone",
  "mermaid", "microphone", "microwave", "monkey", "moon", "mosquito", "motorbike", "mountain",
  "mouse", "moustache", "mouth", "mug", "mushroom", "nail", "necklace", "nose", "octopus",
  "oven", "panda", "pants", "paper clip", "parachute", "parrot", "passport", "peanut", "peas",
  "pencil", "penguin", "piano", "pickup truck", "pig", "pineapple", "pizza", "pool", "popsicle",
  "power outlet", "purse", "rabbit", "raccoon", "radio", "rain", "rainbow", "remote control",
  "rhinoceros", "rifle", "river", "roller coaster", "rollerskates", "sandwich", "saxophone",
  "scissors", "scorpion", "shark", "sheep", "shoe", "shorts", "shovel", "skateboard", "skull",
  "skyscraper", "smiley face", "snail", "snake", "soccer ball", "sock", "spider", "spoon",
  "squirrel", "stairs", "star", "stethoscope", "stop sign", "stove", "strawberry", "submarine",
  "suitcase", "sun", "swan", "sword", "syringe", "t-shirt", "table", "teapot", "teddy-bear",
  "telephone", "television", "tennis racquet", "tent", "tiger", "toaster", "toothbrush",
  "toothpaste", "tornado", "tractor", "traffic light", "train", "tree", "truck", "trumpet",
  "umbrella", "violin", "washing machine", "whale", "wheel", "windmill", "wine bottle",
  "wine glass", "wristwatch", "zebra"
];

const GAME_DURATION = 180; // 3 minutes in seconds

function App() {
  const [gameActive, setGameActive] = useState(false);
  const [gameTimeLeft, setGameTimeLeft] = useState(GAME_DURATION);
  const [currentWord, setCurrentWord] = useState('');
  const [score, setScore] = useState(0);
  const [wordsCompleted, setWordsCompleted] = useState([]);

  useEffect(() => {
    if (!gameActive || gameTimeLeft <= 0) return;

    const timer = setInterval(() => {
      setGameTimeLeft(prev => {
        if (prev <= 1) {
          endGame();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [gameActive, gameTimeLeft]);

  const startGame = () => {
    setGameActive(true);
    setGameTimeLeft(GAME_DURATION);
    setScore(0);
    setWordsCompleted([]);
    pickNewWord([]);
  };

  const endGame = () => {
    setGameActive(false);
  };

  const pickNewWord = (completed) => {
    const available = GAME_CLASSES.filter(word => !completed.includes(word));
    if (available.length === 0) {
      setCurrentWord(GAME_CLASSES[Math.floor(Math.random() * GAME_CLASSES.length)]);
    } else {
      setCurrentWord(available[Math.floor(Math.random() * available.length)]);
    }
  };

  const onCorrectGuess = () => {
    const pointsEarned = Math.ceil(gameTimeLeft / 10);
    setScore(prev => prev + pointsEarned);
    const newCompleted = [...wordsCompleted, currentWord];
    setWordsCompleted(newCompleted);
    
    setTimeout(() => {
      pickNewWord(newCompleted);
    }, 1500);
  };

  return (
    <div className="app">
      <div className="container">
        <div className="left-panel">
          <h1>🎨 Quick Draw Challenge</h1>
          <p className="subtitle">Draw and let AI guess your sketches!</p>
          
          <DrawingCanvasSocket 
            gameActive={gameActive}
            currentWord={currentWord}
            onCorrectGuess={onCorrectGuess}
          />
          
          <GameStatus
            gameActive={gameActive}
            gameTimeLeft={gameTimeLeft}
            currentWord={currentWord}
            score={score}
            wordsCompleted={wordsCompleted}
            onStartGame={startGame}
          />
        </div>
      </div>
    </div>
  );
}

export default App

