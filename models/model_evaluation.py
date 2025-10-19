"""
Model Evaluation Utilities for Furniture Recommendation Platform

This module provides utilities for evaluating AI model performance,
including search quality metrics, relevance scoring, and benchmarking.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    id: str
    title: str
    score: float
    category: str
    price: float
    metadata: Dict[str, Any]

@dataclass
class EvaluationMetrics:
    """Data class for evaluation metrics"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    avg_search_time: float
    total_queries: int
    relevant_results: int

class ModelEvaluator:
    """
    Comprehensive model evaluation for furniture recommendation system
    """
    
    def __init__(self, embedding_model=None, vector_db=None):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.evaluation_history = []
        
    def calculate_precision_at_k(self, 
                                search_results: List[SearchResult], 
                                relevant_items: List[str], 
                                k: int) -> float:
        """
        Calculate Precision@K metric
        
        Args:
            search_results: List of search results
            relevant_items: List of relevant item IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score
        """
        if not search_results or k <= 0:
            return 0.0
            
        top_k_results = search_results[:k]
        relevant_in_top_k = sum(1 for result in top_k_results 
                               if result.id in relevant_items)
        
        return relevant_in_top_k / min(k, len(top_k_results))
    
    def calculate_recall_at_k(self, 
                             search_results: List[SearchResult], 
                             relevant_items: List[str], 
                             k: int) -> float:
        """
        Calculate Recall@K metric
        
        Args:
            search_results: List of search results
            relevant_items: List of relevant item IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if not relevant_items or not search_results or k <= 0:
            return 0.0
            
        top_k_results = search_results[:k]
        relevant_in_top_k = sum(1 for result in top_k_results 
                               if result.id in relevant_items)
        
        return relevant_in_top_k / len(relevant_items)
    
    def calculate_mrr(self, 
                     search_results: List[SearchResult], 
                     relevant_items: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            search_results: List of search results
            relevant_items: List of relevant item IDs
            
        Returns:
            MRR score
        """
        for rank, result in enumerate(search_results, 1):
            if result.id in relevant_items:
                return 1.0 / rank
        return 0.0
    
    def evaluate_query_relevance(self, 
                                query: str, 
                                search_results: List[SearchResult]) -> List[str]:
        """
        Automatically determine relevant items based on query keywords
        
        Args:
            query: Search query
            search_results: List of search results
            
        Returns:
            List of relevant item IDs
        """
        query_words = set(query.lower().split())
        relevant_ids = []
        
        for result in search_results:
            title_words = set(result.title.lower().split())
            category_words = set((result.category or '').lower().split())
            
            # Check for keyword overlap
            if query_words.intersection(title_words.union(category_words)):
                relevant_ids.append(result.id)
        
        return relevant_ids
    
    def benchmark_search_performance(self, 
                                   test_queries: List[str], 
                                   k_values: List[int] = [1, 3, 5, 10]) -> EvaluationMetrics:
        """
        Comprehensive benchmark of search performance
        
        Args:
            test_queries: List of test queries
            k_values: List of K values for precision@k and recall@k
            
        Returns:
            EvaluationMetrics object with comprehensive metrics
        """
        if not self.embedding_model or not self.vector_db:
            raise ValueError("Embedding model and vector database required for benchmarking")
        
        logger.info(f"Starting benchmark with {len(test_queries)} queries")
        
        all_precisions = defaultdict(list)
        all_recalls = defaultdict(list)
        all_mrr_scores = []
        search_times = []
        total_relevant = 0
        
        for query in test_queries:
            logger.debug(f"Processing query: '{query}'")
            
            # Time the search
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            # Perform vector search
            raw_results = self.vector_db.query(query_embedding, top_k=max(k_values))
            
            end_time = time.time()
            search_times.append(end_time - start_time)
            
            # Convert to SearchResult objects
            search_results = []
            for result in raw_results:
                search_results.append(SearchResult(
                    id=result['id'],
                    title=result['metadata'].get('title', ''),
                    score=result['score'],
                    category=result['metadata'].get('category', ''),
                    price=result['metadata'].get('price', 0.0),
                    metadata=result['metadata']
                ))
            
            # Determine relevant items
            relevant_items = self.evaluate_query_relevance(query, search_results)
            total_relevant += len(relevant_items)
            
            # Calculate metrics for different K values
            for k in k_values:
                precision_k = self.calculate_precision_at_k(search_results, relevant_items, k)
                recall_k = self.calculate_recall_at_k(search_results, relevant_items, k)
                
                all_precisions[k].append(precision_k)
                all_recalls[k].append(recall_k)
            
            # Calculate MRR
            mrr_score = self.calculate_mrr(search_results, relevant_items)
            all_mrr_scores.append(mrr_score)
        
        # Calculate average metrics
        avg_precisions = {k: np.mean(scores) for k, scores in all_precisions.items()}
        avg_recalls = {k: np.mean(scores) for k, scores in all_recalls.items()}
        avg_mrr = np.mean(all_mrr_scores)
        avg_search_time = np.mean(search_times)
        
        metrics = EvaluationMetrics(
            precision_at_k=avg_precisions,
            recall_at_k=avg_recalls,
            mrr=avg_mrr,
            avg_search_time=avg_search_time,
            total_queries=len(test_queries),
            relevant_results=total_relevant
        )
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'test_queries': test_queries
        })
        
        logger.info(f"Benchmark complete. Average search time: {avg_search_time:.3f}s")
        logger.info(f"Average MRR: {avg_mrr:.3f}")
        
        return metrics
    
    def compare_models(self, 
                      model_configs: List[Dict[str, Any]], 
                      test_queries: List[str]) -> pd.DataFrame:
        """
        Compare performance of different model configurations
        
        Args:
            model_configs: List of model configuration dictionaries
            test_queries: List of test queries
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for config in model_configs:
            logger.info(f"Evaluating model: {config['name']}")
            
            # Here you would load the specific model configuration
            # For now, we'll use the current model
            metrics = self.benchmark_search_performance(test_queries)
            
            result = {
                'model_name': config['name'],
                'precision_at_1': metrics.precision_at_k.get(1, 0),
                'precision_at_5': metrics.precision_at_k.get(5, 0),
                'precision_at_10': metrics.precision_at_k.get(10, 0),
                'recall_at_5': metrics.recall_at_k.get(5, 0),
                'recall_at_10': metrics.recall_at_k.get(10, 0),
                'mrr': metrics.mrr,
                'avg_search_time': metrics.avg_search_time,
                'total_queries': metrics.total_queries
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, 
                                 metrics: EvaluationMetrics, 
                                 output_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report
        
        Args:
            metrics: EvaluationMetrics object
            output_path: Optional path to save the report
            
        Returns:
            Dictionary with report data
        """
        report = {
            'evaluation_summary': {
                'total_queries': metrics.total_queries,
                'avg_search_time_ms': metrics.avg_search_time * 1000,
                'mean_reciprocal_rank': metrics.mrr,
                'total_relevant_results': metrics.relevant_results
            },
            'precision_metrics': metrics.precision_at_k,
            'recall_metrics': metrics.recall_at_k,
            'performance_analysis': {
                'search_latency': 'Excellent' if metrics.avg_search_time < 0.1 else 
                                'Good' if metrics.avg_search_time < 0.5 else 'Needs Improvement',
                'relevance_quality': 'High' if metrics.mrr > 0.7 else 
                                   'Medium' if metrics.mrr > 0.4 else 'Low',
                'overall_score': self._calculate_overall_score(metrics)
            },
            'recommendations': self._generate_recommendations(metrics),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall model performance score (0-100)"""
        # Weighted combination of different metrics
        precision_score = metrics.precision_at_k.get(5, 0) * 40  # 40% weight
        recall_score = metrics.recall_at_k.get(5, 0) * 30       # 30% weight
        mrr_score = metrics.mrr * 20                             # 20% weight
        speed_score = min(1.0, 0.1 / max(metrics.avg_search_time, 0.001)) * 10  # 10% weight
        
        return precision_score + recall_score + mrr_score + speed_score
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        if metrics.avg_search_time > 0.5:
            recommendations.append("Consider optimizing vector search or using approximate methods")
        
        if metrics.mrr < 0.5:
            recommendations.append("Improve text preprocessing and embedding quality")
        
        avg_precision = np.mean(list(metrics.precision_at_k.values()))
        if avg_precision < 0.3:
            recommendations.append("Consider fine-tuning the embedding model on domain-specific data")
        
        if metrics.total_queries < 50:
            recommendations.append("Increase test dataset size for more reliable evaluation")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory")
        
        return recommendations

class A_BTester:
    """
    A/B testing framework for comparing model performance
    """
    
    def __init__(self):
        self.test_results = []
    
    def run_ab_test(self, 
                   model_a: Any, 
                   model_b: Any, 
                   test_queries: List[str],
                   test_name: str = "AB_Test") -> Dict[str, Any]:
        """
        Run A/B test between two models
        
        Args:
            model_a: First model to test
            model_b: Second model to test
            test_queries: List of test queries
            test_name: Name of the test
            
        Returns:
            Dictionary with A/B test results
        """
        logger.info(f"Starting A/B test: {test_name}")
        
        # Evaluate both models
        evaluator_a = ModelEvaluator(model_a.embedding_model, model_a.vector_db)
        evaluator_b = ModelEvaluator(model_b.embedding_model, model_b.vector_db)
        
        metrics_a = evaluator_a.benchmark_search_performance(test_queries)
        metrics_b = evaluator_b.benchmark_search_performance(test_queries)
        
        # Compare results
        comparison = {
            'test_name': test_name,
            'model_a_metrics': {
                'precision_at_5': metrics_a.precision_at_k.get(5, 0),
                'mrr': metrics_a.mrr,
                'avg_search_time': metrics_a.avg_search_time
            },
            'model_b_metrics': {
                'precision_at_5': metrics_b.precision_at_k.get(5, 0),
                'mrr': metrics_b.mrr,
                'avg_search_time': metrics_b.avg_search_time
            },
            'winner': self._determine_winner(metrics_a, metrics_b),
            'confidence': self._calculate_confidence(metrics_a, metrics_b),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.test_results.append(comparison)
        return comparison
    
    def _determine_winner(self, metrics_a: EvaluationMetrics, metrics_b: EvaluationMetrics) -> str:
        """Determine which model performs better overall"""
        score_a = self._calculate_composite_score(metrics_a)
        score_b = self._calculate_composite_score(metrics_b)
        
        if abs(score_a - score_b) < 0.05:  # Less than 5% difference
            return "Tie"
        elif score_a > score_b:
            return "Model A"
        else:
            return "Model B"
    
    def _calculate_composite_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate composite performance score"""
        precision = metrics.precision_at_k.get(5, 0)
        return precision * 0.7 + metrics.mrr * 0.3  # Weighted score
    
    def _calculate_confidence(self, metrics_a: EvaluationMetrics, metrics_b: EvaluationMetrics) -> float:
        """Calculate statistical confidence in the result"""
        # Simplified confidence calculation
        score_a = self._calculate_composite_score(metrics_a)
        score_b = self._calculate_composite_score(metrics_b)
        
        diff = abs(score_a - score_b)
        return min(0.95, 0.5 + diff)  # Simple confidence based on difference

# Utility functions for quick evaluation
def quick_evaluate(embedding_model, vector_db, test_queries: List[str]) -> Dict[str, float]:
    """
    Quick evaluation with basic metrics
    
    Args:
        embedding_model: Embedding model
        vector_db: Vector database
        test_queries: List of test queries
        
    Returns:
        Dictionary with basic metrics
    """
    evaluator = ModelEvaluator(embedding_model, vector_db)
    metrics = evaluator.benchmark_search_performance(test_queries, k_values=[5])
    
    return {
        'precision_at_5': metrics.precision_at_k.get(5, 0),
        'mrr': metrics.mrr,
        'avg_search_time_ms': metrics.avg_search_time * 1000,
        'total_queries': metrics.total_queries
    }

def load_test_queries(file_path: str) -> List[str]:
    """Load test queries from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning(f"Test queries file not found: {file_path}")
        return [
            "comfortable office chair",
            "dining table for 4 people",
            "storage bedroom furniture",
            "modern sofa grey",
            "wooden coffee table"
        ]

if __name__ == "__main__":
    # Example usage
    logger.basicConfig(level=logging.INFO)
    
    # This would typically be run with actual models
    print("Model evaluation utilities loaded successfully")
    print("Use ModelEvaluator class for comprehensive model evaluation")