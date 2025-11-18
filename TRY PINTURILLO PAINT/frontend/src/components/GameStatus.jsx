const GameStatus = ({ gameActive, gameTimeLeft, currentWord, score, wordsCompleted, onStartGame }) => {
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  if (!gameActive) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        {wordsCompleted.length > 0 ? (
          <>
            <h2 style={{ color: '#667eea' }}>🎉 Game Over!</h2>
            <p style={{ fontSize: '24px', margin: '20px 0' }}>
              <strong>Final Score: {score}</strong>
            </p>
            <p style={{ fontSize: '18px', color: '#666' }}>
              Words completed: {wordsCompleted.length}
            </p>
            <div style={{ 
              marginTop: '15px', 
              maxHeight: '200px', 
              overflowY: 'auto', 
              textAlign: 'left' 
            }}>
              {wordsCompleted.map((word, i) => (
                <div key={i} style={{ margin: '5px 0' }}>
                  {i + 1}. {word}
                </div>
              ))}
            </div>
          </>
        ) : (
          <>
            <h2 style={{ color: '#667eea', marginBottom: '20px' }}>
              🎨 Quick Draw Challenge!
            </h2>
            <p style={{ color: '#666', marginBottom: '30px', lineHeight: '1.6' }}>
              Draw as many objects as you can in 3 minutes!<br />
              The AI will recognize your drawings.<br />
              Faster drawings = more points!
            </p>
          </>
        )}
        <button
          onClick={onStartGame}
          style={{
            marginTop: '20px',
            background: '#667eea',
            color: 'white',
            border: 'none',
            padding: '15px 30px',
            borderRadius: '25px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold'
          }}
        >
          {wordsCompleted.length > 0 ? '🔄 Play Again' : '🚀 Start Game'}
        </button>
      </div>
    );
  }

  return (
    <div
      style={{
        marginTop: '20px',
        padding: '15px',
        borderRadius: '8px',
        background: '#d4edda',
        color: '#155724'
      }}
    >
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '10px'
      }}>
        <div><strong>⏱️ {formatTime(gameTimeLeft)}</strong></div>
        <div><strong>🎯 Draw: {currentWord}</strong></div>
        <div><strong>⭐ Score: {score}</strong></div>
      </div>
    </div>
  );
};

export default GameStatus;
