import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import AnalyticsPage from './pages/AnalyticsPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
        {/* Background Pattern */}
        <div className="fixed inset-0 opacity-5">
          <div className="absolute inset-0" 
               style={{
                 backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
               }}
          />
        </div>

        {/* Main Content */}
        <div className="relative z-10">
          <Navigation />
          
          <motion.main
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="pt-16"
          >
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
            </Routes>
          </motion.main>
        </div>

        {/* Footer */}
        <footer className="relative z-10 mt-20 py-8 bg-slate-900/50 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto px-4 text-center">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
            >
              <p className="text-slate-400 mb-2">
                AI-Powered Furniture Recommendation Platform
              </p>
              <p className="text-slate-500 text-sm">
                Intelligent furniture discovery using semantic search, computer vision, and generative AI
              </p>
            </motion.div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;