import React from 'react';
import './RoundNavigator.css';

export default function RoundNavigator({ currentRound, totalRounds, converged, onSelectRound }) {
  if (!totalRounds || totalRounds <= 1) return null;

  return (
    <div className="round-navigator">
      <div className="round-dots">
        {Array.from({ length: totalRounds }, (_, i) => {
          const roundNum = i + 1;
          const isCompleted = roundNum < currentRound;
          const isActive = roundNum === currentRound;
          return (
            <div
              key={roundNum}
              className={`round-dot ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''} ${onSelectRound ? 'clickable' : ''}`}
              title={`Round ${roundNum}`}
              onClick={() => onSelectRound && onSelectRound(roundNum)}
            />
          );
        })}
      </div>
      <span className="round-label">
        Round {currentRound} of {totalRounds}
        {converged && ` — Converged`}
      </span>
    </div>
  );
}
