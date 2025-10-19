import React, { useState } from 'react';
import apiService from '../utils/apiService';

const APITest = () => {
  const [testResult, setTestResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const testAPI = async () => {
    setIsLoading(true);
    try {
      // Test health check
      const health = await apiService.healthCheck();
      console.log('Health check:', health);
      
      // Test search
      const searchResult = await apiService.searchFurniture('office chair', 'test-session');
      console.log('Search result:', searchResult);
      
      setTestResult(`✅ API Working! Found ${searchResult.results_count} products`);
    } catch (error) {
      console.error('API Test failed:', error);
      setTestResult(`❌ API Failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4 bg-gray-800 rounded-lg m-4">
      <h3 className="text-white text-lg font-semibold mb-2">API Test</h3>
      <button
        onClick={testAPI}
        disabled={isLoading}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {isLoading ? 'Testing...' : 'Test API Connection'}
      </button>
      {testResult && (
        <div className="mt-2 p-2 bg-gray-700 rounded text-white">
          {testResult}
        </div>
      )}
    </div>
  );
};

export default APITest;