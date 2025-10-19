/**
 * Personalization and Context Management Utilities
 * 
 * Handles user preferences, search context, and personalized recommendations
 */

// User preference storage keys
const STORAGE_KEYS = {
  USER_PREFERENCES: 'furniture_user_preferences',
  SEARCH_HISTORY: 'furniture_search_history',
  WISHLIST: 'furniture_wishlist',
  COMPARISON_LIST: 'furniture_comparison',
  USER_CONTEXT: 'furniture_user_context'
};

// Default user preferences
const DEFAULT_PREFERENCES = {
  priceRange: { min: 0, max: 5000 },
  categories: [],
  brands: [],
  materials: [],
  colors: [],
  styles: [],
  rooms: [],
  budget: 'medium',
  priorities: ['price', 'quality', 'style'],
  notifications: true,
  theme: 'dark'
};

/**
 * User Preferences Management
 */
export class UserPreferences {
  constructor() {
    this.preferences = this.loadPreferences();
  }

  loadPreferences() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
      return stored ? { ...DEFAULT_PREFERENCES, ...JSON.parse(stored) } : DEFAULT_PREFERENCES;
    } catch (error) {
      console.error('Error loading user preferences:', error);
      return DEFAULT_PREFERENCES;
    }
  }

  savePreferences(preferences) {
    try {
      this.preferences = { ...this.preferences, ...preferences };
      localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(this.preferences));
      return true;
    } catch (error) {
      console.error('Error saving user preferences:', error);
      return false;
    }
  }

  getPreferences() {
    return { ...this.preferences };
  }

  updatePreference(key, value) {
    return this.savePreferences({ [key]: value });
  }

  resetPreferences() {
    this.preferences = DEFAULT_PREFERENCES;
    localStorage.removeItem(STORAGE_KEYS.USER_PREFERENCES);
    return true;
  }

  // Get personalized search filters based on preferences
  getPersonalizedFilters() {
    return {
      price_min: this.preferences.priceRange.min,
      price_max: this.preferences.priceRange.max,
      categories: this.preferences.categories,
      brands: this.preferences.brands,
      materials: this.preferences.materials,
      colors: this.preferences.colors
    };
  }
}

/**
 * Search History and Context Management
 */
export class SearchContext {
  constructor() {
    this.history = this.loadSearchHistory();
    this.context = this.loadContext();
  }

  loadSearchHistory() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.SEARCH_HISTORY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error loading search history:', error);
      return [];
    }
  }

  loadContext() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.USER_CONTEXT);
      return stored ? JSON.parse(stored) : {
        currentSession: [],
        recentQueries: [],
        preferredCategories: [],
        searchPatterns: {}
      };
    } catch (error) {
      console.error('Error loading user context:', error);
      return {
        currentSession: [],
        recentQueries: [],
        preferredCategories: [],
        searchPatterns: {}
      };
    }
  }

  addSearch(query, results, filters = {}) {
    const searchEntry = {
      id: Date.now(),
      query,
      timestamp: new Date().toISOString(),
      resultsCount: results.length,
      filters,
      clicked: []
    };

    // Add to history (keep last 100 searches)
    this.history.unshift(searchEntry);
    this.history = this.history.slice(0, 100);

    // Update context
    this.context.currentSession.push(searchEntry);
    this.context.recentQueries.unshift(query);
    this.context.recentQueries = [...new Set(this.context.recentQueries)].slice(0, 20);

    // Analyze search patterns
    this.analyzeSearchPatterns(query, filters);

    // Save to storage
    this.saveSearchHistory();
    this.saveContext();

    return searchEntry;
  }

  analyzeSearchPatterns(query, filters) {
    // Extract keywords and patterns
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    
    words.forEach(word => {
      this.context.searchPatterns[word] = (this.context.searchPatterns[word] || 0) + 1;
    });

    // Track category preferences
    if (filters.categories && filters.categories.length > 0) {
      filters.categories.forEach(category => {
        const index = this.context.preferredCategories.findIndex(c => c.name === category);
        if (index >= 0) {
          this.context.preferredCategories[index].count++;
        } else {
          this.context.preferredCategories.push({ name: category, count: 1 });
        }
      });
      
      // Sort by frequency
      this.context.preferredCategories.sort((a, b) => b.count - a.count);
    }
  }

  getSearchSuggestions(currentQuery = '') {
    const suggestions = [];
    
    // Recent queries that match current input
    const recentMatches = this.context.recentQueries
      .filter(q => q.toLowerCase().includes(currentQuery.toLowerCase()))
      .slice(0, 5);

    suggestions.push(...recentMatches.map(query => ({
      type: 'recent',
      text: query,
      source: 'recent searches'
    })));

    // Popular search patterns
    const popularTerms = Object.entries(this.context.searchPatterns)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .filter(([term]) => term.includes(currentQuery.toLowerCase()))
      .map(([term]) => ({
        type: 'popular',
        text: term,
        source: 'popular searches'
      }));

    suggestions.push(...popularTerms);

    return suggestions.slice(0, 8);
  }

  getContextualFilters() {
    // Suggest filters based on user's search history
    const filters = {};

    // Most searched categories
    if (this.context.preferredCategories.length > 0) {
      filters.suggestedCategories = this.context.preferredCategories.slice(0, 3);
    }

    // Price range based on search history
    const priceRanges = this.history
      .map(h => h.filters.price_min || 0)
      .filter(p => p > 0);
    
    if (priceRanges.length > 0) {
      filters.suggestedPriceRange = {
        min: Math.min(...priceRanges),
        max: Math.max(...priceRanges) || 1000
      };
    }

    return filters;
  }

  saveSearchHistory() {
    try {
      localStorage.setItem(STORAGE_KEYS.SEARCH_HISTORY, JSON.stringify(this.history));
    } catch (error) {
      console.error('Error saving search history:', error);
    }
  }

  saveContext() {
    try {
      localStorage.setItem(STORAGE_KEYS.USER_CONTEXT, JSON.stringify(this.context));
    } catch (error) {
      console.error('Error saving user context:', error);
    }
  }

  clearHistory() {
    this.history = [];
    this.context = {
      currentSession: [],
      recentQueries: [],
      preferredCategories: [],
      searchPatterns: {}
    };
    localStorage.removeItem(STORAGE_KEYS.SEARCH_HISTORY);
    localStorage.removeItem(STORAGE_KEYS.USER_CONTEXT);
  }
}

/**
 * Wishlist Management
 */
export class WishlistManager {
  constructor() {
    this.wishlist = this.loadWishlist();
  }

  loadWishlist() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.WISHLIST);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error loading wishlist:', error);
      return [];
    }
  }

  saveWishlist() {
    try {
      localStorage.setItem(STORAGE_KEYS.WISHLIST, JSON.stringify(this.wishlist));
      return true;
    } catch (error) {
      console.error('Error saving wishlist:', error);
      return false;
    }
  }

  addItem(product) {
    // Prevent duplicates
    if (!this.isInWishlist(product.id)) {
      this.wishlist.push({
        ...product,
        addedAt: new Date().toISOString(),
        notes: ''
      });
      this.saveWishlist();
      return true;
    }
    return false;
  }

  removeItem(productId) {
    const initialLength = this.wishlist.length;
    this.wishlist = this.wishlist.filter(item => item.id !== productId);
    
    if (this.wishlist.length < initialLength) {
      this.saveWishlist();
      return true;
    }
    return false;
  }

  updateNotes(productId, notes) {
    const item = this.wishlist.find(item => item.id === productId);
    if (item) {
      item.notes = notes;
      this.saveWishlist();
      return true;
    }
    return false;
  }

  isInWishlist(productId) {
    return this.wishlist.some(item => item.id === productId);
  }

  getWishlist() {
    return [...this.wishlist];
  }

  getWishlistCount() {
    return this.wishlist.length;
  }

  clearWishlist() {
    this.wishlist = [];
    localStorage.removeItem(STORAGE_KEYS.WISHLIST);
    return true;
  }

  // Get wishlist items by category
  getByCategory(category) {
    return this.wishlist.filter(item => 
      item.category && item.category.toLowerCase().includes(category.toLowerCase())
    );
  }

  // Get recently added items
  getRecentlyAdded(limit = 5) {
    return this.wishlist
      .sort((a, b) => new Date(b.addedAt) - new Date(a.addedAt))
      .slice(0, limit);
  }
}

/**
 * Product Comparison Management
 */
export class ComparisonManager {
  constructor() {
    this.comparison = this.loadComparison();
    this.maxItems = 4; // Maximum items to compare
  }

  loadComparison() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.COMPARISON_LIST);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error loading comparison list:', error);
      return [];
    }
  }

  saveComparison() {
    try {
      localStorage.setItem(STORAGE_KEYS.COMPARISON_LIST, JSON.stringify(this.comparison));
      return true;
    } catch (error) {
      console.error('Error saving comparison list:', error);
      return false;
    }
  }

  addItem(product) {
    // Check if already in comparison
    if (this.isInComparison(product.id)) {
      return { success: false, message: 'Item already in comparison' };
    }

    // Check if we've reached the limit
    if (this.comparison.length >= this.maxItems) {
      return { success: false, message: `Maximum ${this.maxItems} items for comparison` };
    }

    this.comparison.push({
      ...product,
      addedAt: new Date().toISOString()
    });
    
    this.saveComparison();
    return { success: true, message: 'Item added to comparison' };
  }

  removeItem(productId) {
    const initialLength = this.comparison.length;
    this.comparison = this.comparison.filter(item => item.id !== productId);
    
    if (this.comparison.length < initialLength) {
      this.saveComparison();
      return true;
    }
    return false;
  }

  isInComparison(productId) {
    return this.comparison.some(item => item.id === productId);
  }

  getComparison() {
    return [...this.comparison];
  }

  getComparisonCount() {
    return this.comparison.length;
  }

  clearComparison() {
    this.comparison = [];
    localStorage.removeItem(STORAGE_KEYS.COMPARISON_LIST);
    return true;
  }

  // Generate comparison matrix
  getComparisonMatrix() {
    if (this.comparison.length < 2) {
      return null;
    }

    const features = ['title', 'price', 'category', 'brand', 'material', 'color', 'rating'];
    const matrix = {
      products: this.comparison,
      features: features,
      data: {}
    };

    features.forEach(feature => {
      matrix.data[feature] = this.comparison.map(product => product[feature] || 'N/A');
    });

    return matrix;
  }

  // Get similar products recommendations based on comparison items
  getSimilarRecommendations() {
    if (this.comparison.length === 0) return [];

    // Extract common features from comparison items
    const categories = [...new Set(this.comparison.map(item => item.category).filter(Boolean))];
    const brands = [...new Set(this.comparison.map(item => item.brand).filter(Boolean))];
    const materials = [...new Set(this.comparison.map(item => item.material).filter(Boolean))];

    return {
      suggestedFilters: {
        categories,
        brands,
        materials
      },
      searchQuery: `${categories.join(' or ')} furniture ${brands.join(' ')}`
    };
  }
}

/**
 * Personalization Engine
 */
export class PersonalizationEngine {
  constructor() {
    this.preferences = new UserPreferences();
    this.searchContext = new SearchContext();
    this.wishlist = new WishlistManager();
    this.comparison = new ComparisonManager();
  }

  // Get personalized product recommendations
  getPersonalizedRecommendations(products = []) {
    const userPrefs = this.preferences.getPreferences();
    const context = this.searchContext.context;

    // Score products based on user preferences
    const scoredProducts = products.map(product => {
      let score = 0;

      // Price preference scoring
      const price = parseFloat(product.price) || 0;
      if (price >= userPrefs.priceRange.min && price <= userPrefs.priceRange.max) {
        score += 10;
      }

      // Category preference scoring
      if (userPrefs.categories.includes(product.category)) {
        score += 15;
      }

      // Brand preference scoring
      if (userPrefs.brands.includes(product.brand)) {
        score += 10;
      }

      // Search history scoring
      const categoryFreq = context.preferredCategories.find(c => c.name === product.category);
      if (categoryFreq) {
        score += Math.min(categoryFreq.count * 2, 20);
      }

      // Wishlist affinity scoring
      const wishlistItems = this.wishlist.getWishlist();
      const similarInWishlist = wishlistItems.filter(item => 
        item.category === product.category || item.brand === product.brand
      );
      score += similarInWishlist.length * 5;

      return { ...product, personalizedScore: score };
    });

    // Sort by personalized score
    return scoredProducts.sort((a, b) => b.personalizedScore - a.personalizedScore);
  }

  // Get smart search suggestions based on context
  getSmartSuggestions(currentQuery = '') {
    const suggestions = this.searchContext.getSearchSuggestions(currentQuery);
    const contextFilters = this.searchContext.getContextualFilters();

    return {
      querySuggestions: suggestions,
      filterSuggestions: contextFilters,
      personalizedCategories: this.preferences.preferences.categories
    };
  }

  // Generate insights for user dashboard
  generateUserInsights() {
    const preferences = this.preferences.getPreferences();
    const searchHistory = this.searchContext.history;
    const wishlistCount = this.wishlist.getWishlistCount();
    const comparisonCount = this.comparison.getComparisonCount();

    return {
      profile: {
        preferredBudget: preferences.budget,
        favoriteCategories: preferences.categories.slice(0, 3),
        searchActivity: searchHistory.length,
        wishlistSize: wishlistCount,
        activeComparisons: comparisonCount
      },
      activity: {
        totalSearches: searchHistory.length,
        recentSearches: searchHistory.slice(0, 5),
        topSearchTerms: Object.entries(this.searchContext.context.searchPatterns)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 5)
          .map(([term, count]) => ({ term, count }))
      },
      recommendations: {
        suggestedQueries: this.searchContext.getSearchSuggestions().slice(0, 3),
        recommendedCategories: this.searchContext.context.preferredCategories.slice(0, 3)
      }
    };
  }

  // Export user data
  exportUserData() {
    return {
      preferences: this.preferences.getPreferences(),
      searchHistory: this.searchContext.history,
      wishlist: this.wishlist.getWishlist(),
      comparison: this.comparison.getComparison(),
      context: this.searchContext.context,
      exportedAt: new Date().toISOString()
    };
  }

  // Import user data
  importUserData(userData) {
    try {
      if (userData.preferences) {
        this.preferences.savePreferences(userData.preferences);
      }
      if (userData.wishlist) {
        localStorage.setItem(STORAGE_KEYS.WISHLIST, JSON.stringify(userData.wishlist));
        this.wishlist = new WishlistManager();
      }
      if (userData.comparison) {
        localStorage.setItem(STORAGE_KEYS.COMPARISON_LIST, JSON.stringify(userData.comparison));
        this.comparison = new ComparisonManager();
      }
      return { success: true, message: 'User data imported successfully' };
    } catch (error) {
      console.error('Error importing user data:', error);
      return { success: false, message: 'Failed to import user data' };
    }
  }

  // Clear all user data
  clearAllData() {
    this.preferences.resetPreferences();
    this.searchContext.clearHistory();
    this.wishlist.clearWishlist();
    this.comparison.clearComparison();
    return true;
  }
}

// Export singleton instances
export const personalization = new PersonalizationEngine();
export const userPreferences = personalization.preferences;
export const searchContext = personalization.searchContext;
export const wishlist = personalization.wishlist;
export const comparison = personalization.comparison;

// Export for direct class usage
export {
  UserPreferences,
  SearchContext,
  WishlistManager,
  ComparisonManager,
  PersonalizationEngine
};