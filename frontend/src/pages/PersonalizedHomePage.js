import React, { useState, useEffect, useMemo } from 'react';
import { Search, Heart, BarChart3, ShoppingCart, Star, Filter, Sparkles, TrendingUp } from 'lucide-react';
import ProductCard from '../components/ProductCard';
import LoadingSpinner from '../components/LoadingSpinner';
import apiService from '../utils/apiService';
import { personalization, wishlist, comparison, searchContext } from '../utils/personalizations';

const PersonalizedHomePage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [userInsights, setUserInsights] = useState(null);
  const [showPersonalized, setShowPersonalized] = useState(true);
  const [searchSuggestions, setSearchSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [wishlistCount, setWishlistCount] = useState(0);
  const [comparisonCount, setComparisonCount] = useState(0);

  // Initialize personalization data
  useEffect(() => {
    const initializePersonalization = () => {
      // Load user insights
      const insights = personalization.generateUserInsights();
      setUserInsights(insights);
      
      // Update counts
      setWishlistCount(wishlist.getWishlistCount());
      setComparisonCount(comparison.getComparisonCount());
      
      // Initial welcome message with personalization
      const welcomeMessage = {
        id: 1,
        type: 'bot',
        text: insights.profile.favoriteCategories.length > 0 
          ? `Welcome back! Based on your preferences for ${insights.profile.favoriteCategories.join(', ')}, I can help you find the perfect furniture. What are you looking for today?`
          : "Welcome to your personalized furniture assistant! I'll help you discover amazing furniture and learn your preferences along the way. What can I help you find?",
        timestamp: new Date(),
        isPersonalized: true
      };
      
      setMessages([welcomeMessage]);
    };

    initializePersonalization();
  }, []);

  // Get smart search suggestions
  const getSmartSuggestions = (query) => {
    const suggestions = personalization.getSmartSuggestions(query);
    setSearchSuggestions([
      ...suggestions.querySuggestions,
      ...suggestions.personalizedCategories.map(cat => ({
        type: 'category',
        text: cat,
        source: 'your preferences'
      }))
    ]);
  };

  // Handle input change with suggestions
  const handleInputChange = (e) => {
    const value = e.target.value;
    setInputValue(value);
    
    if (value.length > 1) {
      getSmartSuggestions(value);
      setShowSuggestions(true);
    } else {
      setShowSuggestions(false);
    }
  };

  // Handle search submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    setIsLoading(true);
    setShowSuggestions(false);

    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputValue,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      // Get personalized filters
      const personalizedFilters = personalization.preferences.getPersonalizedFilters();
      
      // Perform search with personalization
      const response = await apiService.searchFurniture(inputValue, personalizedFilters);
      
      if (response.success && response.data?.results) {
        // Add search to context
        const searchEntry = searchContext.addSearch(inputValue, response.data.results, personalizedFilters);
        
        // Get personalized recommendations
        const personalizedResults = showPersonalized 
          ? personalization.getPersonalizedRecommendations(response.data.results)
          : response.data.results;

        setSearchResults(personalizedResults);
        
        // Create bot response
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          text: `Found ${personalizedResults.length} ${showPersonalized ? 'personalized ' : ''}results for "${inputValue}". ${showPersonalized ? 'Results are ranked based on your preferences!' : ''}`,
          timestamp: new Date(),
          searchResults: personalizedResults,
          isPersonalized: showPersonalized,
          searchId: searchEntry.id
        };
        setMessages(prev => [...prev, botMessage]);
        
        // Update insights
        const newInsights = personalization.generateUserInsights();
        setUserInsights(newInsights);
      } else {
        throw new Error(response.message || 'Search failed');
      }
    } catch (error) {
      console.error('Search error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        text: "I'm having trouble searching right now. Let me show you some popular furniture options instead!",
        timestamp: new Date(),
        searchResults: getSampleProducts()
      };
      setMessages(prev => [...prev, errorMessage]);
      setSearchResults(getSampleProducts());
    }

    setInputValue('');
    setIsLoading(false);
  };

  // Handle suggestion selection
  const handleSuggestionSelect = (suggestion) => {
    setInputValue(suggestion.text);
    setShowSuggestions(false);
    // Auto-submit the suggestion
    setTimeout(() => {
      const form = document.querySelector('form');
      if (form) {
        form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }, 100);
  };

  // Handle wishlist actions
  const handleWishlistToggle = (product) => {
    if (wishlist.isInWishlist(product.id)) {
      wishlist.removeItem(product.id);
      setWishlistCount(prev => prev - 1);
    } else {
      wishlist.addItem(product);
      setWishlistCount(prev => prev + 1);
    }
  };

  // Handle comparison actions
  const handleComparisonToggle = (product) => {
    if (comparison.isInComparison(product.id)) {
      comparison.removeItem(product.id);
      setComparisonCount(prev => prev - 1);
    } else {
      const result = comparison.addItem(product);
      if (result.success) {
        setComparisonCount(prev => prev + 1);
      } else {
        alert(result.message);
      }
    }
  };

  // Sample products for fallback
  const getSampleProducts = () => [
    {
      id: 'sample-1',
      title: 'Modern Ergonomic Office Chair',
      price: 299.99,
      category: 'Office Furniture',
      brand: 'ComfortCorp',
      material: 'Fabric',
      color: 'Black',
      rating: 4.5,
      image: 'https://via.placeholder.com/300x200?text=Office+Chair'
    },
    {
      id: 'sample-2',
      title: 'Scandinavian Dining Table',
      price: 599.99,
      category: 'Dining Room',
      brand: 'Nordic Design',
      material: 'Oak Wood',
      color: 'Natural',
      rating: 4.8,
      image: 'https://via.placeholder.com/300x200?text=Dining+Table'
    },
    {
      id: 'sample-3',
      title: 'Luxury Velvet Sofa',
      price: 1299.99,
      category: 'Living Room',
      brand: 'Elegance',
      material: 'Velvet',
      color: 'Navy',
      rating: 4.7,
      image: 'https://via.placeholder.com/300x200?text=Velvet+Sofa'
    }
  ];

  // Personalized recommendations based on user data
  const personalizedRecommendations = useMemo(() => {
    if (!userInsights) return [];
    
    const sampleProducts = getSampleProducts();
    return personalization.getPersonalizedRecommendations(sampleProducts).slice(0, 3);
  }, [userInsights]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-indigo-900">
      {/* Header with Stats */}
      <div className="bg-gray-800/50 backdrop-blur-sm border-b border-gray-700">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <h1 className="text-2xl font-bold text-white flex items-center">
              <Sparkles className="mr-2 text-yellow-400" />
              Personalized Furniture Discovery
            </h1>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Personalization Toggle */}
            <button
              onClick={() => setShowPersonalized(!showPersonalized)}
              className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all duration-200 ${
                showPersonalized
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span>{showPersonalized ? 'Personalized' : 'Standard'}</span>
            </button>

            {/* Stats */}
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-1 text-pink-300">
                <Heart className="w-4 h-4" />
                <span>{wishlistCount}</span>
              </div>
              <div className="flex items-center space-x-1 text-blue-300">
                <BarChart3 className="w-4 h-4" />
                <span>{comparisonCount}/4</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* User Insights Panel */}
      {userInsights && userInsights.profile.favoriteCategories.length > 0 && (
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl p-4 border border-purple-500/20">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium mb-1">Your Preferences</h3>
                <div className="flex flex-wrap gap-2">
                  {userInsights.profile.favoriteCategories.map((category, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-purple-600/20 text-purple-200 rounded-full text-sm border border-purple-500/30"
                    >
                      {category}
                    </span>
                  ))}
                </div>
              </div>
              <div className="text-right text-sm text-gray-300">
                <div>{userInsights.profile.searchActivity} searches</div>
                <div>{userInsights.profile.wishlistSize} wishlist items</div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Chat Interface */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl border border-gray-700">
              
              {/* Chat Messages */}
              <div className="h-96 overflow-y-auto p-6 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                        message.type === 'user'
                          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                          : message.isPersonalized
                            ? 'bg-gradient-to-r from-purple-900/50 to-blue-900/50 text-gray-100 border border-purple-500/30'
                            : 'bg-gray-700 text-gray-100'
                      }`}
                    >
                      <p className="text-sm">{message.text}</p>
                      {message.isPersonalized && (
                        <div className="mt-2 flex items-center text-xs text-purple-300">
                          <Sparkles className="w-3 h-3 mr-1" />
                          Personalized for you
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-700 rounded-2xl px-4 py-3">
                      <LoadingSpinner />
                    </div>
                  </div>
                )}
              </div>

              {/* Search Input */}
              <div className="p-6 border-t border-gray-700 relative">
                <form onSubmit={handleSubmit} className="flex space-x-2">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputValue}
                      onChange={handleInputChange}
                      placeholder="Describe the furniture you're looking for..."
                      className="w-full px-4 py-3 bg-gray-700 text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 border border-gray-600"
                      disabled={isLoading}
                    />
                    
                    {/* Search Suggestions */}
                    {showSuggestions && searchSuggestions.length > 0 && (
                      <div className="absolute top-full left-0 right-0 bg-gray-800 border border-gray-600 rounded-xl mt-1 max-h-60 overflow-y-auto z-50 shadow-xl">
                        {searchSuggestions.map((suggestion, index) => (
                          <button
                            key={index}
                            type="button"
                            onClick={() => handleSuggestionSelect(suggestion)}
                            className="w-full text-left px-4 py-3 hover:bg-gray-700 text-gray-200 border-b border-gray-700 last:border-b-0 flex items-center justify-between"
                          >
                            <span>{suggestion.text}</span>
                            <span className={`text-xs px-2 py-1 rounded ${
                              suggestion.type === 'recent' ? 'bg-blue-900/30 text-blue-300' :
                              suggestion.type === 'popular' ? 'bg-green-900/30 text-green-300' :
                              'bg-purple-900/30 text-purple-300'
                            }`}>
                              {suggestion.source}
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                  >
                    <Search className="w-5 h-5" />
                  </button>
                </form>
              </div>
            </div>
          </div>

          {/* Sidebar with Recommendations */}
          <div className="space-y-6">
            
            {/* Personalized Recommendations */}
            {personalizedRecommendations.length > 0 && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                  <Sparkles className="mr-2 text-purple-400" />
                  For You
                </h3>
                <div className="space-y-4">
                  {personalizedRecommendations.map((product) => (
                    <div key={product.id} className="bg-gray-700/50 rounded-lg p-3 border border-gray-600">
                      <div className="flex items-center space-x-3">
                        <div className="w-12 h-12 bg-gray-600 rounded-lg flex-shrink-0">
                          {/* Placeholder for product image */}
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="text-sm font-medium text-white truncate">{product.title}</h4>
                          <p className="text-sm text-purple-300">${product.price}</p>
                          {product.personalizedScore > 0 && (
                            <div className="flex items-center text-xs text-yellow-400">
                              <Star className="w-3 h-3 mr-1 fill-current" />
                              {Math.round(product.personalizedScore)}% match
                            </div>
                          )}
                        </div>
                        <div className="flex space-x-1">
                          <button
                            onClick={() => handleWishlistToggle(product)}
                            className={`p-1 rounded ${
                              wishlist.isInWishlist(product.id)
                                ? 'text-pink-400 hover:text-pink-300'
                                : 'text-gray-400 hover:text-pink-400'
                            }`}
                          >
                            <Heart className={`w-4 h-4 ${wishlist.isInWishlist(product.id) ? 'fill-current' : ''}`} />
                          </button>
                          <button
                            onClick={() => handleComparisonToggle(product)}
                            className={`p-1 rounded ${
                              comparison.isInComparison(product.id)
                                ? 'text-blue-400 hover:text-blue-300'
                                : 'text-gray-400 hover:text-blue-400'
                            }`}
                          >
                            <BarChart3 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quick Actions */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button className="w-full text-left p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg transition-all duration-200 text-gray-200 border border-gray-600">
                  <div className="flex items-center">
                    <Heart className="w-5 h-5 mr-3 text-pink-400" />
                    <div>
                      <div className="font-medium">View Wishlist</div>
                      <div className="text-sm text-gray-400">{wishlistCount} items</div>
                    </div>
                  </div>
                </button>
                
                <button className="w-full text-left p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg transition-all duration-200 text-gray-200 border border-gray-600">
                  <div className="flex items-center">
                    <BarChart3 className="w-5 h-5 mr-3 text-blue-400" />
                    <div>
                      <div className="font-medium">Compare Products</div>
                      <div className="text-sm text-gray-400">{comparisonCount}/4 selected</div>
                    </div>
                  </div>
                </button>
                
                <button className="w-full text-left p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg transition-all duration-200 text-gray-200 border border-gray-600">
                  <div className="flex items-center">
                    <Filter className="w-5 h-5 mr-3 text-green-400" />
                    <div>
                      <div className="font-medium">Preferences</div>
                      <div className="text-sm text-gray-400">Customize settings</div>
                    </div>
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="mt-12">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-2xl font-bold text-white">
                {showPersonalized ? 'Personalized Results' : 'Search Results'}
              </h2>
              {showPersonalized && (
                <div className="text-sm text-purple-300 flex items-center">
                  <Sparkles className="w-4 h-4 mr-1" />
                  Ranked by your preferences
                </div>
              )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {searchResults.map((product) => (
                <div key={product.id} className="relative">
                  <ProductCard
                    product={product}
                    onWishlistToggle={() => handleWishlistToggle(product)}
                    onCompareToggle={() => handleComparisonToggle(product)}
                    isInWishlist={wishlist.isInWishlist(product.id)}
                    isInComparison={comparison.isInComparison(product.id)}
                  />
                  {product.personalizedScore > 15 && (
                    <div className="absolute -top-2 -right-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white text-xs px-2 py-1 rounded-full flex items-center">
                      <Star className="w-3 h-3 mr-1 fill-current" />
                      Top Pick
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PersonalizedHomePage;