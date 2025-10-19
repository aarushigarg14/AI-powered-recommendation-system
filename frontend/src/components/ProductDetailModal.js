import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Heart, 
  ShoppingCart, 
  Star, 
  Info, 
  Package, 
  Truck, 
  Shield,
  ChevronLeft,
  ChevronRight,
  MapPin,
  Tag,
  Ruler
} from 'lucide-react';
import HighlightText from './HighlightText';

const ProductDetailModal = ({ product, isOpen, onClose, onAddToWishlist, onAddToCart, isInWishlist, searchTerms = [] }) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [quantity, setQuantity] = useState(1);

  if (!product || !isOpen) return null;

  const {
    id,
    title,
    price,
    category,
    images = [],
    description,
    original_description,
    material,
    color,
    brand,
    manufacturer,
    country_of_origin,
    package_dimensions,
    categories = [],
    similarity_score
  } = product;

  // Handle both formats - ensure we have images array
  const productImages = images.length > 0 
    ? images 
    : ['https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&h=600&fit=crop'];

  const handleImageError = (e) => {
    e.target.src = 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&h=600&fit=crop';
  };

  const nextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % productImages.length);
  };

  const prevImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + productImages.length) % productImages.length);
  };

  const handleAddToCart = () => {
    for (let i = 0; i < quantity; i++) {
      onAddToCart(product);
    }
  };

  const renderStars = (rating) => {
    const stars = [];
    const numRating = rating || 4.2; // Default rating for demo
    const fullStars = Math.floor(numRating);
    const hasHalfStar = numRating % 1 !== 0;

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(
          <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
        );
      } else if (i === fullStars && hasHalfStar) {
        stars.push(
          <div key={i} className="relative">
            <Star className="h-5 w-5 text-gray-600" />
            <div className="absolute inset-0 overflow-hidden" style={{ width: '50%' }}>
              <Star className="h-5 w-5 text-yellow-400 fill-current" />
            </div>
          </div>
        );
      } else {
        stars.push(
          <Star key={i} className="h-5 w-5 text-gray-600" />
        );
      }
    }
    return stars;
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.3 }}
          className="bg-slate-900 rounded-2xl border border-slate-700 max-w-6xl w-full max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-700">
            <h2 className="text-2xl font-bold text-white">Product Details</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-800 rounded-full transition-colors"
            >
              <X className="h-6 w-6 text-slate-400" />
            </button>
          </div>

          <div className="overflow-y-auto max-h-[calc(90vh-80px)]">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-6">
              {/* Product Images */}
              <div className="space-y-4">
                {/* Main Image */}
                <div className="relative aspect-square bg-slate-800 rounded-xl overflow-hidden">
                  <img
                    src={productImages[currentImageIndex]}
                    alt={title}
                    className="w-full h-full object-cover"
                    onError={handleImageError}
                  />
                  
                  {/* Image Navigation */}
                  {productImages.length > 1 && (
                    <>
                      <button
                        onClick={prevImage}
                        className="absolute left-4 top-1/2 -translate-y-1/2 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
                      >
                        <ChevronLeft className="h-6 w-6" />
                      </button>
                      <button
                        onClick={nextImage}
                        className="absolute right-4 top-1/2 -translate-y-1/2 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
                      >
                        <ChevronRight className="h-6 w-6" />
                      </button>
                    </>
                  )}

                  {/* Image Counter */}
                  {productImages.length > 1 && (
                    <div className="absolute bottom-4 right-4 px-3 py-1 bg-black/70 rounded-full text-white text-sm">
                      {currentImageIndex + 1} / {productImages.length}
                    </div>
                  )}
                </div>

                {/* Thumbnail Images */}
                {productImages.length > 1 && (
                  <div className="flex space-x-2 overflow-x-auto">
                    {productImages.map((image, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentImageIndex(index)}
                        className={`flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border-2 transition-colors ${
                          index === currentImageIndex ? 'border-blue-500' : 'border-slate-600'
                        }`}
                      >
                        <img
                          src={image}
                          alt={`${title} ${index + 1}`}
                          className="w-full h-full object-cover"
                          onError={handleImageError}
                        />
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Product Information */}
              <div className="space-y-6">
                {/* Title and Category */}
                <div>
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="px-3 py-1 bg-blue-500 text-white text-sm font-medium rounded-full">
                      {category}
                    </span>
                    {similarity_score && (
                      <span className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded-full">
                        {similarity_score.toFixed(1)}% match
                      </span>
                    )}
                  </div>
                  <h1 className="text-3xl font-bold text-white mb-2">
                    <HighlightText 
                      text={title} 
                      searchTerms={searchTerms}
                      className=""
                    />
                  </h1>
                  
                  {/* Rating */}
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center space-x-1">
                      {renderStars(4.2)}
                    </div>
                    <span className="text-slate-400">4.2 (128 reviews)</span>
                  </div>
                </div>

                {/* Price */}
                <div className="text-4xl font-bold text-blue-400">
                  ${price?.toFixed(2)}
                </div>

                {/* Description */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-white">Description</h3>
                  <p className="text-slate-300 leading-relaxed">
                    <HighlightText 
                      text={original_description || description} 
                      searchTerms={searchTerms}
                      className=""
                    />
                  </p>
                </div>

                {/* Product Details Grid */}
                <div className="grid grid-cols-2 gap-4">
                  {brand && (
                    <div className="flex items-center space-x-2 text-slate-300">
                      <Tag className="h-5 w-5 text-slate-400" />
                      <div>
                        <span className="text-slate-400 text-sm">Brand</span>
                        <p className="font-medium">{brand}</p>
                      </div>
                    </div>
                  )}

                  {material && (
                    <div className="flex items-center space-x-2 text-slate-300">
                      <Info className="h-5 w-5 text-slate-400" />
                      <div>
                        <span className="text-slate-400 text-sm">Material</span>
                        <p className="font-medium">{material}</p>
                      </div>
                    </div>
                  )}

                  {color && (
                    <div className="flex items-center space-x-2 text-slate-300">
                      <div 
                        className="w-5 h-5 rounded-full border border-slate-600"
                        style={{ backgroundColor: getColorCode(color) }}
                      />
                      <div>
                        <span className="text-slate-400 text-sm">Color</span>
                        <p className="font-medium">{color}</p>
                      </div>
                    </div>
                  )}

                  {manufacturer && (
                    <div className="flex items-center space-x-2 text-slate-300">
                      <Package className="h-5 w-5 text-slate-400" />
                      <div>
                        <span className="text-slate-400 text-sm">Manufacturer</span>
                        <p className="font-medium">{manufacturer}</p>
                      </div>
                    </div>
                  )}

                  {country_of_origin && (
                    <div className="flex items-center space-x-2 text-slate-300">
                      <MapPin className="h-5 w-5 text-slate-400" />
                      <div>
                        <span className="text-slate-400 text-sm">Origin</span>
                        <p className="font-medium">{country_of_origin}</p>
                      </div>
                    </div>
                  )}

                  {package_dimensions && (
                    <div className="flex items-center space-x-2 text-slate-300">
                      <Ruler className="h-5 w-5 text-slate-400" />
                      <div>
                        <span className="text-slate-400 text-sm">Dimensions</span>
                        <p className="font-medium">{package_dimensions}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Categories */}
                {categories.length > 0 && (
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-2">Categories</h3>
                    <div className="flex flex-wrap gap-2">
                      {categories.map((cat, index) => (
                        <span
                          key={index}
                          className="px-3 py-1 bg-slate-800 text-slate-300 text-sm rounded-full"
                        >
                          {cat}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Quantity and Actions */}
                <div className="space-y-4 pt-4 border-t border-slate-700">
                  {/* Quantity Selector */}
                  <div className="flex items-center space-x-4">
                    <span className="text-slate-300">Quantity:</span>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setQuantity(Math.max(1, quantity - 1))}
                        className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-white transition-colors"
                      >
                        -
                      </button>
                      <span className="px-4 py-2 bg-slate-800 rounded-lg text-white min-w-12 text-center">
                        {quantity}
                      </span>
                      <button
                        onClick={() => setQuantity(quantity + 1)}
                        className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-white transition-colors"
                      >
                        +
                      </button>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-4">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => onAddToWishlist(product)}
                      className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                        isInWishlist
                          ? 'bg-pink-500 hover:bg-pink-600 text-white'
                          : 'bg-slate-800 hover:bg-pink-500 text-slate-300 hover:text-white'
                      }`}
                    >
                      <Heart className={`h-5 w-5 ${isInWishlist ? 'fill-current' : ''}`} />
                      <span>{isInWishlist ? 'In Wishlist' : 'Add to Wishlist'}</span>
                    </motion.button>

                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleAddToCart}
                      className="flex items-center space-x-2 px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors duration-200"
                    >
                      <ShoppingCart className="h-5 w-5" />
                      <span>Add to Cart</span>
                    </motion.button>
                  </div>

                  {/* Additional Info */}
                  <div className="flex items-center justify-between text-sm text-slate-400 pt-4">
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-1">
                        <Truck className="h-4 w-4" />
                        <span>Free shipping</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Shield className="h-4 w-4" />
                        <span>1 year warranty</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
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

export default ProductDetailModal;