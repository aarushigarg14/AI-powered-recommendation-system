import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles, Heart, ShoppingCart, Eye, Star } from 'lucide-react';
import ProductCard from '../components/ProductCard';
import ProductDetailModal from '../components/ProductDetailModal';
import LoadingSpinner from '../components/LoadingSpinner';
import APITest from '../components/APITest';
import apiService from '../utils/apiService';

const HomePage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const [wishlist, setWishlist] = useState([]);
  const [cart, setCart] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentSearchQuery, setCurrentSearchQuery] = useState('');
  const messagesEndRef = useRef(null);

  // Sample welcome message
  useEffect(() => {
    setMessages([
      {
        id: 1,
        type: 'ai',
        content: `🏠 Welcome to AI Furniture Discovery! I'm your intelligent furniture assistant.

I can help you find the perfect furniture using natural language. Try asking me:
• "Show me modern tables under $200"
• "I need a comfortable grey sofa for my living room"
• "Find wooden dining tables that seat 6 people"
• "What storage solutions do you have?"

What furniture are you looking for today?`,
        timestamp: new Date(),
        suggestions: [
          "Modern sofa under $300",
          "Wooden dining table",
          "Storage solutions",
          "Office chair ergonomic"
        ]
      }
    ]);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentSearchQuery(inputValue.trim()); // Store current search query for highlighting
    setInputValue('');
    setIsLoading(true);

    try {
      // Call real backend API
      const response = await apiService.searchFurniture(inputValue.trim(), sessionId);
      
      if (response.success) {
        const aiMessage = {
          id: Date.now() + 1,
          type: 'ai',
          content: response.message,
          products: response.results || response.data?.results || [],
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, aiMessage]);
      } else {
        throw new Error(response.message || 'Search failed');
      }
    } catch (error) {
      console.error('Search failed:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: `I apologize, but I'm having trouble searching right now. Here are some sample furniture items that might interest you:`,
        products: generateSampleProducts(),
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInputValue(suggestion);
  };

  const addToWishlist = (product) => {
    setWishlist(prev => {
      if (prev.find(item => item.id === product.id)) {
        return prev.filter(item => item.id !== product.id);
      }
      return [...prev, product];
    });
  };

  const addToCart = (product) => {
    setCart(prev => [...prev, product]);
  };

  const viewProduct = (product) => {
    setSelectedProduct(product);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedProduct(null);
  };

  const isInWishlist = (productId) => wishlist.some(item => item.id === productId);

  // Extract search terms from query for highlighting
  const extractSearchTerms = (query) => {
    if (!query) return [];
    
    // Remove price constraints and other filters from query
    let cleanQuery = query.toLowerCase();
    
    // Remove price patterns
    const pricePatterns = [
      /\bunder\s*\$?\d+(?:\.\d{2})?\b/g,
      /\bbelow\s*\$?\d+(?:\.\d{2})?\b/g,
      /\bless\s+than\s*\$?\d+(?:\.\d{2})?\b/g,
      /\bup\s+to\s*\$?\d+(?:\.\d{2})?\b/g,
      /\bover\s*\$?\d+(?:\.\d{2})?\b/g,
      /\babove\s*\$?\d+(?:\.\d{2})?\b/g,
      /\bmore\s+than\s*\$?\d+(?:\.\d{2})?\b/g,
      /\bbetween\s*\$?\d+(?:\.\d{2})?\s*and\s*\$?\d+(?:\.\d{2})?\b/g
    ];
    
    pricePatterns.forEach(pattern => {
      cleanQuery = cleanQuery.replace(pattern, '');
    });
    
    // Remove relevance patterns
    const relevancePatterns = [
      /\b(?:with|under|having)\s+(?:low|poor|bad)\s+relevance\b/g,
      /\b(?:with|under|having)\s+(?:high|good|strong)\s+relevance\b/g,
      /\b(?:low|poor|bad)\s+relevance\b/g,
      /\b(?:high|good|strong)\s+relevance\b/g
    ];
    
    relevancePatterns.forEach(pattern => {
      cleanQuery = cleanQuery.replace(pattern, '');
    });
    
    // Split into words and filter out empty strings and common words
    const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'need', 'show', 'find', 'get', 'give', 'take', 'make', 'look', 'want', 'like', 'go', 'come', 'see', 'know', 'think', 'say', 'tell', 'ask', 'use', 'work', 'try', 'keep', 'let', 'put', 'end', 'why', 'how', 'where', 'when', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs'];
    
    return cleanQuery
      .split(/\s+/)
      .map(term => term.trim())
      .filter(term => term.length >= 2 && !stopWords.includes(term))
      .filter(term => !/^\d+$/.test(term)); // Remove pure numbers
  };

  // Generate sample products for fallback
  const generateSampleProducts = () => {
    return [
      {
        id: 'fallback-1',
        title: 'Modern Ergonomic Office Chair',
        price: 299.99,
        category: 'Office Furniture',
        images: ['https://images.unsplash.com/photo-1541558869434-2840d308329a?w=400'],
        description: 'Experience ultimate comfort with this modern ergonomic office chair.',
        material: 'Fabric',
        color: 'Black',
        rating: 4.5
      },
      {
        id: 'fallback-2',
        title: 'Scandinavian Dining Table',
        price: 599.99,
        category: 'Dining Room',
        images: ['https://images.unsplash.com/photo-1449247709967-d4461a6a6103?w=400'],
        description: 'Beautifully crafted oak dining table perfect for family gatherings.',
        material: 'Oak Wood',
        color: 'Natural',
        rating: 4.8
      },
      {
        id: 'fallback-3',
        title: 'Luxury Velvet Sofa',
        price: 1299.99,
        category: 'Living Room',
        images: ['https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400'],
        description: 'Indulge in luxury with this premium velvet sectional sofa.',
        material: 'Velvet',
        color: 'Navy Blue',
        rating: 4.7
      }
    ];
  };

  return (
    <div className="min-h-screen pt-20 pb-8">
      <div className="max-w-6xl mx-auto px-4">
        <APITest />
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            AI-Powered <span className="text-gradient">Furniture Discovery</span>
          </h1>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto">
            Chat with our intelligent assistant to find the perfect furniture using natural language
          </p>
        </motion.div>

        {/* Chat Container */}
        <div className="bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 shadow-2xl overflow-hidden">
          {/* Chat Messages */}
          <div className="h-96 md:h-[500px] overflow-y-auto chat-container p-4 space-y-4">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className={`flex items-start space-x-3 ${
                    message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}
                >
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.type === 'user' 
                      ? 'bg-blue-600' 
                      : 'bg-gradient-to-br from-indigo-500 to-purple-600'
                  }`}>
                    {message.type === 'user' ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>

                  {/* Message Content */}
                  <div className={`flex-1 ${message.type === 'user' ? 'text-right' : ''}`}>
                    <div className={`chat-bubble ${message.type}`}>
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      
                      {/* Suggestions */}
                      {message.suggestions && (
                        <div className="mt-4 flex flex-wrap gap-2">
                          {message.suggestions.map((suggestion, index) => (
                            <button
                              key={index}
                              onClick={() => handleSuggestionClick(suggestion)}
                              className="btn-secondary text-sm py-1 px-3 hover:bg-blue-600/20 hover:border-blue-500/50"
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Products Grid */}
                      {message.products && message.products.length > 0 && (
                        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                          {message.products.map((product) => (
                            <ProductCard
                              key={product.id}
                              product={product}
                              onAddToWishlist={() => addToWishlist(product)}
                              onAddToCart={() => addToCart(product)}
                              onViewProduct={() => viewProduct(product)}
                              isInWishlist={isInWishlist(product.id)}
                              searchTerms={extractSearchTerms(currentSearchQuery)}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                    
                    <div className="text-xs text-slate-400 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Loading Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start space-x-3"
              >
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-white" />
                </div>
                <div className="chat-bubble ai">
                  <LoadingSpinner />
                  <span className="ml-3 text-slate-300">Searching for furniture...</span>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <div className="border-t border-slate-700/50 p-4">
            <form onSubmit={handleSubmit} className="flex items-center space-x-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask about furniture... e.g., 'Show me modern sofas under $500'"
                  className="chat-input pr-12"
                  disabled={isLoading}
                />
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <Sparkles className="h-5 w-5 text-slate-400" />
                </div>
              </div>
              
              <motion.button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                <Send className="h-5 w-5" />
                <span>Send</span>
              </motion.button>
            </form>
          </div>
        </div>

        {/* Stats Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <div className="bg-slate-800/50 rounded-xl p-4 text-center backdrop-blur-sm border border-slate-700/50">
            <div className="text-2xl font-bold text-blue-400">{cart.length}</div>
            <div className="text-sm text-slate-400">In Cart</div>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 text-center backdrop-blur-sm border border-slate-700/50">
            <div className="text-2xl font-bold text-pink-400">{wishlist.length}</div>
            <div className="text-sm text-slate-400">Wishlist</div>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 text-center backdrop-blur-sm border border-slate-700/50">
            <div className="text-2xl font-bold text-green-400">312</div>
            <div className="text-sm text-slate-400">Products</div>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 text-center backdrop-blur-sm border border-slate-700/50">
            <div className="text-2xl font-bold text-purple-400">25+</div>
            <div className="text-sm text-slate-400">Categories</div>
          </div>
        </motion.div>
      </div>
      
      {/* Product Detail Modal */}
      <ProductDetailModal
        product={selectedProduct}
        isOpen={isModalOpen}
        onClose={closeModal}
        onAddToWishlist={addToWishlist}
        onAddToCart={addToCart}
        isInWishlist={selectedProduct ? isInWishlist(selectedProduct.id) : false}
        searchTerms={extractSearchTerms(currentSearchQuery)}
      />
    </div>
  );
};

export default HomePage;
