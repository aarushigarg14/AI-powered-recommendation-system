import React from 'react';
import { motion } from 'framer-motion';
import { Heart, ShoppingCart, Eye, Star, Info } from 'lucide-react';
import HighlightText from './HighlightText';

const ProductCard = ({ product, onAddToWishlist, onAddToCart, onViewProduct, isInWishlist, searchTerms = [] }) => {
  const {
    id,
    title,
    price,
    category,
    image,
    images,
    description,
    material,
    color,
    rating = 0
  } = product;
  
  // Handle both formats - backend sends 'images' array, frontend might use 'image'
  const productImage = image || (images && images[0]) || 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400&h=300&fit=crop';

  const handleImageError = (e) => {
    e.target.src = 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400&h=300&fit=crop';
  };

  const renderStars = (rating) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(
          <Star key={i} className="h-4 w-4 text-yellow-400 fill-current" />
        );
      } else if (i === fullStars && hasHalfStar) {
        stars.push(
          <div key={i} className="relative">
            <Star className="h-4 w-4 text-gray-600" />
            <div className="absolute inset-0 overflow-hidden" style={{ width: '50%' }}>
              <Star className="h-4 w-4 text-yellow-400 fill-current" />
            </div>
          </div>
        );
      } else {
        stars.push(
          <Star key={i} className="h-4 w-4 text-gray-600" />
        );
      }
    }
    return stars;
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.3 }}
      className="product-card group"
    >
      {/* Product Image */}
      <div className="relative overflow-hidden rounded-lg mb-4">
        <img
          src={productImage}
          alt={title}
          className="product-image"
          onError={handleImageError}
          loading="lazy"
        />
        
        {/* Overlay Actions */}
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 
                        transition-opacity duration-300 flex items-center justify-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onAddToWishlist(product)}
            className={`p-2 rounded-full backdrop-blur-sm transition-colors duration-200 ${
              isInWishlist 
                ? 'bg-pink-500 text-white' 
                : 'bg-white/20 text-white hover:bg-pink-500'
            }`}
            title={isInWishlist ? "Remove from wishlist" : "Add to wishlist"}
          >
            <Heart className={`h-5 w-5 ${isInWishlist ? 'fill-current' : ''}`} />
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onAddToCart(product)}
            className="p-2 rounded-full bg-blue-500 text-white hover:bg-blue-600 
                       backdrop-blur-sm transition-colors duration-200"
            title="Add to cart"
          >
            <ShoppingCart className="h-5 w-5" />
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onViewProduct(product)}
            className="p-2 rounded-full bg-white/20 text-white hover:bg-slate-600 
                       backdrop-blur-sm transition-colors duration-200"
            title="View details"
          >
            <Eye className="h-5 w-5" />
          </motion.button>
        </div>

        {/* Category Badge */}
        <div className="absolute top-3 left-3">
          <span className="px-2 py-1 bg-blue-500/90 text-white text-xs font-medium 
                          rounded-full backdrop-blur-sm">
            {category}
          </span>
        </div>

        {/* Price Badge */}
        <div className="absolute top-3 right-3">
          <span className="px-3 py-1 bg-slate-900/90 text-white font-bold text-sm 
                          rounded-full backdrop-blur-sm">
            ${price?.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Product Info */}
      <div className="space-y-3">
        {/* Title */}
        <h3 className="text-lg font-semibold text-white group-hover:text-blue-400 
                      transition-colors duration-200 line-clamp-2">
          <HighlightText 
            text={title} 
            searchTerms={searchTerms}
            className=""
          />
        </h3>

        {/* Rating */}
        {rating > 0 && (
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              {renderStars(rating)}
            </div>
            <span className="text-sm text-slate-400">({rating})</span>
          </div>
        )}

        {/* Description */}
        <p className="text-slate-300 text-sm line-clamp-2 leading-relaxed">
          <HighlightText 
            text={description} 
            searchTerms={searchTerms}
            className=""
          />
        </p>

        {/* Material & Color */}
        <div className="flex items-center space-x-4 text-xs text-slate-400">
          {material && (
            <div className="flex items-center space-x-1">
              <Info className="h-3 w-3" />
              <span>{material}</span>
            </div>
          )}
          {color && (
            <div className="flex items-center space-x-1">
              <div 
                className="w-3 h-3 rounded-full border border-slate-600"
                style={{ backgroundColor: getColorCode(color) }}
              />
              <span>{color}</span>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-2">
          <div className="text-xl font-bold text-blue-400">
            ${price?.toFixed(2)}
          </div>
          
          <div className="flex items-center space-x-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onAddToWishlist(product)}
              className={`p-2 rounded-lg transition-colors duration-200 ${
                isInWishlist
                  ? 'bg-pink-500 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-pink-500 hover:text-white'
              }`}
            >
              <Heart className={`h-4 w-4 ${isInWishlist ? 'fill-current' : ''}`} />
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onAddToCart(product)}
              className="btn-primary py-2 px-4 text-sm"
            >
              Add to Cart
            </motion.button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Helper function to get color code for display
const getColorCode = (colorName) => {
  const colorMap = {
    'red': '#ef4444',
    'blue': '#3b82f6',
    'green': '#10b981',
    'yellow': '#f59e0b',
    'orange': '#f97316',
    'purple': '#8b5cf6',
    'pink': '#ec4899',
    'brown': '#a3724a',
    'black': '#1f2937',
    'white': '#f8fafc',
    'gray': '#6b7280',
    'grey': '#6b7280',
    'beige': '#d4c5a9',
    'cream': '#f5f5dc',
    'navy': '#1e40af',
    'gold': '#f59e0b',
    'silver': '#9ca3af'
  };
  
  return colorMap[colorName?.toLowerCase()] || '#6b7280';
};

export default ProductCard;