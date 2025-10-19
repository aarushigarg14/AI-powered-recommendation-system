import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens or common headers here
    console.log(`Making API request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`API response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('Response error:', error.response?.data || error.message);
    
    // Handle specific error cases
    if (error.response?.status === 404) {
      console.error('API endpoint not found');
    } else if (error.response?.status === 500) {
      console.error('Server error occurred');
    } else if (error.code === 'NETWORK_ERROR') {
      console.error('Network error - server may be offline');
    }
    
    return Promise.reject(error);
  }
);

// API service methods
const apiService = {
  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Search for furniture
  async searchFurniture(query, sessionId, filters = null, maxResults = 20) {
    try {
      const requestBody = {
        query,
        session_id: sessionId,
        max_results: maxResults
      };
      
      if (filters) {
        requestBody.filters = filters;
      }
      
      const response = await api.post('/api/search', requestBody);
      return response.data;
    } catch (error) {
      console.error('Search failed:', error);
      throw error;
    }
  },

  // Get analytics data
  async getAnalytics() {
    try {
      const response = await api.get('/api/analytics');
      return response.data;
    } catch (error) {
      console.error('Analytics fetch failed:', error);
      throw error;
    }
  },

  // Get detailed server status
  async getStatus() {
    try {
      const response = await api.get('/api/status');
      return response.data;
    } catch (error) {
      console.error('Status check failed:', error);
      throw error;
    }
  },

  // Add to wishlist (future implementation)
  async addToWishlist(productId, userId) {
    try {
      const response = await api.post('/api/wishlist', {
        product_id: productId,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      console.error('Add to wishlist failed:', error);
      throw error;
    }
  },

  // Add to cart (future implementation)
  async addToCart(productId, userId, quantity = 1) {
    try {
      const response = await api.post('/api/cart', {
        product_id: productId,
        user_id: userId,
        quantity
      });
      return response.data;
    } catch (error) {
      console.error('Add to cart failed:', error);
      throw error;
    }
  },

  // Get product details (future implementation)
  async getProductDetails(productId) {
    try {
      const response = await api.get(`/api/products/${productId}`);
      return response.data;
    } catch (error) {
      console.error('Get product details failed:', error);
      throw error;
    }
  }
};

export default apiService;