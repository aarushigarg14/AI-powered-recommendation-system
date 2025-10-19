import React from 'react';

const HighlightText = ({ text, searchTerms, className = "" }) => {
  if (!text || !searchTerms || searchTerms.length === 0) {
    return <span className={className}>{text}</span>;
  }

  // Clean and prepare search terms
  const cleanSearchTerms = searchTerms
    .filter(term => term && term.trim().length > 0)
    .map(term => term.trim().toLowerCase())
    .filter(term => term.length >= 2); // Only highlight terms with 2+ characters

  if (cleanSearchTerms.length === 0) {
    return <span className={className}>{text}</span>;
  }

  // Create regex pattern for all search terms
  const pattern = cleanSearchTerms
    .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) // Escape special regex characters
    .join('|');
  
  const regex = new RegExp(`(${pattern})`, 'gi');
  
  // Split text by search terms while preserving the terms
  const parts = text.split(regex);
  
  return (
    <span className={className}>
      {parts.map((part, index) => {
        const isMatch = cleanSearchTerms.some(term => 
          part.toLowerCase() === term.toLowerCase()
        );
        
        return isMatch ? (
          <span
            key={index}
            className="bg-yellow-300 text-yellow-900 px-1 py-0.5 rounded font-semibold"
            style={{
              backgroundColor: '#fef08a',
              color: '#713f12',
              boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
            }}
          >
            {part}
          </span>
        ) : (
          <span key={index}>{part}</span>
        );
      })}
    </span>
  );
};

export default HighlightText;