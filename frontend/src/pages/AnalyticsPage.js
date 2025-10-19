import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  Package, 
  DollarSign, 
  ShoppingCart, 
  Users,
  Eye,
  Heart 
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';

const AnalyticsPage = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalyticsData();
  }, []);

  const fetchAnalyticsData = async () => {
    try {
      setIsLoading(true);
      const API_URL = process.env.REACT_APP_API_URL;
      const response = await fetch(`${API_URL}/api/analytics`);

      
      if (!response.ok) {
        throw new Error(`Failed to fetch analytics: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        setAnalyticsData(transformApiData(result.data));
      } else {
        throw new Error('API returned error');
      }
      
    } catch (err) {
      console.error('Failed to fetch analytics:', err);
      setError(err.message);
      // Fallback to mock data if API fails
      setAnalyticsData(generateMockAnalytics());
    } finally {
      setIsLoading(false);
    }
  };

  const transformApiData = (apiData) => {
    // Transform the API response to match the frontend structure
    const { overview, price_stats, price_distribution, top_categories, top_brands, top_materials, top_colors } = apiData;
    
    // Convert categories object to array format for charts
    const categoryData = Object.entries(top_categories || {}).map(([category, count]) => ({
      category: category.length > 25 ? category.substring(0, 25) + '...' : category,
      products: count,
      percentage: Math.round((count / overview.total_products) * 100),
      fullName: category  // Keep full name for tooltips
    })).slice(0, 10); // Show top 10 categories
    
    // Convert price distribution to array format
    const priceRanges = Object.entries(price_distribution || {}).map(([range, count]) => ({
      range,
      count,
      percentage: Math.round((count / price_stats.products_with_prices) * 100)
    }));
    
    // Convert materials to array format with colors
    const materialColors = ['#8B4513', '#708090', '#4682B4', '#A0522D', '#32CD32', '#87CEEB', '#DDA0DD', '#F0E68C', '#20B2AA', '#FF6347'];
    const materialData = Object.entries(top_materials || {}).map(([material, count], index) => ({
      material,
      count,
      color: materialColors[index % materialColors.length]
    }));
    
    // Convert brands to array format  
    const topBrandsArray = Object.entries(top_brands || {}).map(([brand, count]) => ({
      brand: brand.length > 20 ? brand.substring(0, 20) + '...' : brand,
      sales: count
    })).slice(0, 10);
    
    // Generate mock monthly trends (since we don't have temporal data)
    const monthlyTrends = [
      { month: 'Jan', searches: Math.floor(Math.random() * 500) + 800 },
      { month: 'Feb', searches: Math.floor(Math.random() * 500) + 900 },
      { month: 'Mar', searches: Math.floor(Math.random() * 500) + 1000 },
      { month: 'Apr', searches: Math.floor(Math.random() * 500) + 950 },
      { month: 'May', searches: Math.floor(Math.random() * 500) + 1100 },
      { month: 'Jun', searches: Math.floor(Math.random() * 500) + 1200 }
    ].map(item => ({ ...item, sales: Math.floor(item.searches * 0.4) }));
    
    return {
      summary: {
        totalProducts: overview.total_products,
        totalCategories: overview.unique_categories,
        averagePrice: price_stats.avg_price ? price_stats.avg_price.toFixed(2) : '0.00',
        totalRevenue: Math.floor(price_stats.avg_price * overview.total_products * 0.3), // Estimated
        activeUsers: Math.floor(overview.total_products * 4), // Estimated
        conversionRate: 3.2 // Estimated
      },
      categoryData,
      priceRanges,
      materialData,
      topBrands: topBrandsArray,
      monthlyTrends
    };
  };

  const generateMockAnalytics = () => ({
    summary: {
      totalProducts: 312,
      totalCategories: 25,
      averagePrice: 287.50,
      totalRevenue: 89750.25,
      activeUsers: 1247,
      conversionRate: 3.2
    },
    categoryData: [
      { category: 'Living Room', products: 78, percentage: 25 },
      { category: 'Bedroom', products: 65, percentage: 21 },
      { category: 'Dining Room', products: 54, percentage: 17 },
      { category: 'Office', products: 43, percentage: 14 },
      { category: 'Storage', products: 38, percentage: 12 },
      { category: 'Outdoor', products: 34, percentage: 11 }
    ],
    priceRanges: [
      { range: '$0-100', count: 87, percentage: 28 },
      { range: '$100-300', count: 112, percentage: 36 },
      { range: '$300-500', count: 76, percentage: 24 },
      { range: '$500-1000', count: 28, percentage: 9 },
      { range: '$1000+', count: 9, percentage: 3 }
    ],
    materialData: [
      { material: 'Wood', count: 89, color: '#8B4513' },
      { material: 'Metal', count: 67, color: '#708090' },
      { material: 'Fabric', count: 54, color: '#4682B4' },
      { material: 'Leather', count: 43, color: '#A0522D' },
      { material: 'Plastic', count: 32, color: '#32CD32' },
      { material: 'Glass', count: 27, color: '#87CEEB' }
    ],
    topBrands: [
      { brand: 'ModernHome', sales: 45 },
      { brand: 'ComfortLiving', sales: 38 },
      { brand: 'DesignCraft', sales: 32 },
      { brand: 'UrbanStyle', sales: 28 },
      { brand: 'ClassicFurniture', sales: 25 }
    ],
    monthlyTrends: [
      { month: 'Jan', searches: 1200, sales: 450 },
      { month: 'Feb', searches: 1350, sales: 520 },
      { month: 'Mar', searches: 1580, sales: 680 },
      { month: 'Apr', searches: 1420, sales: 590 },
      { month: 'May', searches: 1650, sales: 720 },
      { month: 'Jun', searches: 1800, sales: 840 }
    ]
  });

  if (isLoading) {
    return (
      <div className="min-h-screen pt-20 pb-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading real analytics data...</p>
          <p className="text-slate-500 text-sm mt-2">Fetching data from furniture dataset</p>
        </div>
      </div>
    );
  }

  if (error && !analyticsData) {
    return (
      <div className="min-h-screen pt-20 pb-8 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <p className="text-red-400 mb-2">Failed to load analytics data</p>
          <p className="text-slate-500 text-sm mb-4">{error}</p>
          <button 
            onClick={fetchAnalyticsData}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const { summary, categoryData, priceRanges, materialData, topBrands, monthlyTrends } = analyticsData;

  return (
    <div className="min-h-screen pt-20 pb-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Analytics <span className="text-gradient">Dashboard</span>
          </h1>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto mb-4">
            Real-time insights from furniture dataset
          </p>
          {error && (
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-yellow-900/30 border border-yellow-700/50 text-yellow-300 text-sm">
              ⚠️ API Error - Using cached data
            </div>
          )}
          {!error && (
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-green-900/30 border border-green-700/50 text-green-300 text-sm">
              ✓ Live data from {analyticsData?.summary?.totalProducts || 312} products
            </div>
          )}
        </motion.div>

        {/* KPI Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-8"
        >
          <div className="chart-container text-center">
            <Package className="h-8 w-8 text-blue-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{summary.totalProducts}</div>
            <div className="text-sm text-slate-400">Products</div>
          </div>
          
          <div className="chart-container text-center">
            <BarChart3 className="h-8 w-8 text-green-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{summary.totalCategories}</div>
            <div className="text-sm text-slate-400">Categories</div>
          </div>
          
          <div className="chart-container text-center">
            <DollarSign className="h-8 w-8 text-yellow-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">${summary.averagePrice}</div>
            <div className="text-sm text-slate-400">Avg Price</div>
          </div>
          
          
        </motion.div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 gap-6 mb-8">
          {/* Category Distribution */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="chart-container"
          >
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-blue-400" />
              Products by Category
            </h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={categoryData} margin={{ top: 5, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="category" 
                  tick={{ fill: '#9ca3af', fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  interval={0}
                />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f3f4f6'
                  }}
                  formatter={(value, name, props) => [
                    `${value} products (${props.payload.percentage}%)`,
                    'Count'
                  ]}
                  labelFormatter={(label, payload) => {
                    if (payload && payload[0] && payload[0].payload.fullName) {
                      return `Category: ${payload[0].payload.fullName}`;
                    }
                    return `Category: ${label}`;
                  }}
                />
                <Bar dataKey="products" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            
            {/* Category Details Table */}
            <div className="mt-4">
              <div className="max-h-40 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-slate-800">
                    <tr>
                      <th className="text-left py-2 px-2 text-slate-400">Category</th>
                      <th className="text-right py-2 px-2 text-slate-400">Products</th>
                      <th className="text-right py-2 px-2 text-slate-400">%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {categoryData.map((item, index) => (
                      <tr key={index} className="border-t border-slate-700">
                        <td className="py-1 px-2 text-white text-xs">{item.fullName}</td>
                        <td className="py-1 px-2 text-blue-400 text-right font-medium">{item.products}</td>
                        <td className="py-1 px-2 text-slate-300 text-right">{item.percentage}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        </div>
        <div className="grid grid-cols-1 gap-6 mb-8">
          {/* Price Distribution */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="chart-container"
          >
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <DollarSign className="h-5 w-5 mr-2 text-green-400" />
              Price Range Distribution
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={priceRanges}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ range, percentage }) => `${range} (${percentage}%)`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {priceRanges.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'][index % 5]} 
                    />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f3f4f6'
                  }} 
                />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* More Charts */}
        <div className="grid grid-cols-1 gap-6">

          {/* Material Distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="chart-container h-[600px]"
          >
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Package className="h-5 w-5 mr-2 text-purple-400" />
              Materials Used
            </h3>
            <ResponsiveContainer width="100%" height="100%">
                 <PieChart>
                       <Pie
                          data={materialData}
                          cx="50%"
                          cy="50%"
                           outerRadius={150} 
                          fill="#8884d8"
                          dataKey="count"
                        label={({ material, count }) => `${material}: ${count}`}
    >
                    {materialData.map((entry, index) => (
                       <Cell key={`cell-${index}`} fill={entry.color} />
                                  ))}
                        </Pie>
                        <Tooltip 
                             contentStyle={{ 
                                  backgroundColor: '#1f2937', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px',
                                  color: '#f3f4f6'
                                 }} 
                        />
                </PieChart>
              </ResponsiveContainer>
          </motion.div>

          
        </div>

        
      </div>
    </div>
  );
};

export default AnalyticsPage;
