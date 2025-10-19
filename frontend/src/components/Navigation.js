import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Home, BarChart3, Bot, Sparkles } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    {
      path: '/',
      name: 'Product Discovery',
      icon: Bot,
      description: 'AI-powered furniture search'
    },
    {
      path: '/analytics',
      name: 'Analytics',
      icon: BarChart3,
      description: 'Business insights dashboard'
    }
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-lg border-b border-slate-700/50"
    >
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            to="/"
            className="flex items-center space-x-3 text-white hover:text-blue-400 transition-colors duration-200"
          >
            <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg">
              <Sparkles className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">AI Furniture</h1>
              <p className="text-xs text-slate-400 -mt-1">Smart Discovery</p>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className="relative group"
                >
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`
                      flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200
                      ${isActive(item.path)
                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                      }
                    `}
                  >
                    <Icon className="h-5 w-5" />
                    <span className="font-medium">{item.name}</span>
                  </motion.div>

                  {/* Tooltip */}
                  <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 
                                  opacity-0 group-hover:opacity-100 transition-opacity duration-200
                                  bg-slate-800 text-white text-sm px-3 py-2 rounded-lg shadow-lg
                                  pointer-events-none whitespace-nowrap z-50">
                    {item.description}
                    <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 
                                    bg-slate-800 rotate-45"></div>
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-slate-400">Online</span>
            </div>
          </div>
        </div>
      </div>

      {/* Active Tab Indicator */}
      <motion.div
        className="absolute bottom-0 h-0.5 bg-gradient-to-r from-blue-500 to-indigo-500"
        initial={false}
        animate={{
          x: navItems.findIndex(item => isActive(item.path)) * 200 + 100,
          width: 120
        }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      />
    </motion.nav>
  );
};

export default Navigation;