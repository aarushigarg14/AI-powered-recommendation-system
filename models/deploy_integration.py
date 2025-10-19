"""
Deployment Integration Script for AI Models

This script provides utilities to integrate trained AI models
into the production FastAPI backend server.
"""

import os
import sys
import json
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class ModelDeploymentManager:
    """
    Manages the deployment and integration of trained AI models
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.config = None
        self.deployment_status = {}
        
    def validate_model_artifacts(self) -> Dict[str, bool]:
        """
        Validate that all required model artifacts exist
        
        Returns:
            Dictionary with validation results
        """
        required_files = [
            "model_config.json",
            "product_embeddings.npy",
            "vector_db.pkl",
            "product_metadata.pkl"
        ]
        
        validation_results = {}
        
        for file_name in required_files:
            file_path = self.models_dir / file_name
            exists = file_path.exists()
            validation_results[file_name] = exists
            
            if exists:
                # Check file size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ {file_name}: {size_mb:.2f} MB")
            else:
                logger.error(f"❌ Missing: {file_name}")
        
        all_valid = all(validation_results.values())
        logger.info(f"Model artifacts validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
        
        return validation_results
    
    def load_model_config(self) -> Dict[str, Any]:
        """
        Load model configuration from artifacts
        
        Returns:
            Model configuration dictionary
        """
        config_path = self.models_dir / "model_config.json"
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info(f" Loaded model config: {self.config.get('embedding_model', 'Unknown')}")
            logger.info(f" Model metrics: {self.config.get('performance_metrics', {})}")
            
            return self.config
            
        except Exception as e:
            logger.error(f" Failed to load model config: {str(e)}")
            raise
    
    def generate_backend_integration_code(self, output_path: str = None) -> str:
        """
        Generate integration code for the FastAPI backend
        
        Args:
            output_path: Optional path to save the integration code
            
        Returns:
            Integration code as string
        """
        if not self.config:
            self.load_model_config()
        
        integration_code = f'''"""
AI Models Integration for FastAPI Backend
Generated automatically by ModelDeploymentManager

Integration Instructions:
1. Copy this code to your backend/ai_models.py file
2. Update the AIModelManager class with the load_pretrained_models method
3. Call await ai_manager.load_pretrained_models() in main.py startup
"""

import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AIModelManager:
    """Enhanced AI Model Manager with pre-trained model support"""
    
    def __init__(self, settings):
        self.settings = settings
        self.embedding_model = None
        self.genai_model = None
        self.genai_tokenizer = None
        self.vector_db = {{"vectors": {{}}, "metadata": {{}}}}
        self.product_metadata = []
        self.model_config = None
        
    async def load_pretrained_models(self) -> bool:
        """
        Load pre-trained models from training artifacts
        
        Returns:
            True if successful, False otherwise
        """
        try:
            models_dir = Path("models")
            
            # Load model configuration
            config_path = models_dir / "model_config.json"
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            
            logger.info(f" Loading model: {{self.model_config['embedding_model']}}")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_config['embedding_model'])
            logger.info(f" Embedding model loaded ({{self.model_config['embedding_dimension']}} dimensions)")
            
            # Load GenAI model if available
            if self.model_config.get('genai_model'):
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    self.genai_tokenizer = AutoTokenizer.from_pretrained(self.model_config['genai_model'])
                    self.genai_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config['genai_model'])
                    logger.info(" Generative AI model loaded")
                except Exception as e:
                    logger.warning(f" Could not load GenAI model: {{str(e)}}")
            
            # Load vector database
            vector_db_path = models_dir / "vector_db.pkl"
            with open(vector_db_path, 'rb') as f:
                db_data = pickle.load(f)
                self.vector_db = db_data
            
            logger.info(f" Vector database loaded: {{len(self.vector_db['vectors'])}} products")
            
            # Load product metadata
            metadata_path = models_dir / "product_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                self.product_metadata = pickle.load(f)
            
            logger.info(f" Product metadata loaded: {{len(self.product_metadata)}} products")
            
            # Log performance metrics
            if 'performance_metrics' in self.model_config:
                metrics = self.model_config['performance_metrics']
                logger.info(f" Model Performance:")
                logger.info(f"   - Precision: {{metrics.get('precision', 0):.3f}}")
                logger.info(f"   - Search Time: {{metrics.get('avg_search_time', 0)*1000:.1f}}ms")
            
            return True
            
        except Exception as e:
            logger.error(f" Failed to load pre-trained models: {{str(e)}}")
            return False
    
    def semantic_search(self, 
                       query: str, 
                       top_k: int = 10, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search using pre-trained embeddings
        """
        if not self.embedding_model or not self.vector_db:
            logger.error("Models not loaded. Call load_pretrained_models() first.")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            # Search vector database
            results = self._search_vectors(query_embedding, top_k)
            
            # Apply filters if provided
            if filters:
                results = self._apply_filters(results, filters)
            
            # Format results
            formatted_results = []
            for result in results[:top_k]:
                metadata = result['metadata']
                formatted_result = {{
                    'id': result['id'],
                    'title': metadata.get('title', ''),
                    'price': metadata.get('price'),
                    'category': metadata.get('category'),
                    'brand': metadata.get('brand'),
                    'material': metadata.get('material'),
                    'color': metadata.get('color'),
                    'similarity_score': float(result['score']),
                    'ai_description': self.generate_description(metadata)
                }}
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {{str(e)}}")
            return []
    
    def _search_vectors(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search vectors using cosine similarity"""
        if not self.vector_db['vectors']:
            return []
        
        similarities = []
        for product_id, vector in self.vector_db['vectors'].items():
            similarity = cosine_similarity([query_embedding], [vector])[0][0]
            similarities.append({{
                'id': product_id,
                'score': float(similarity),
                'metadata': self.vector_db['metadata'][product_id]
            }})
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            metadata = result['metadata']
            include_result = True
            
            # Price range filter
            if 'price_min' in filters or 'price_max' in filters:
                price = metadata.get('price', 0)
                if 'price_min' in filters and price < filters['price_min']:
                    include_result = False
                if 'price_max' in filters and price > filters['price_max']:
                    include_result = False
            
            # Category filter
            if 'categories' in filters:
                result_category = metadata.get('category', '').lower()
                filter_categories = [cat.lower() for cat in filters['categories']]
                if not any(cat in result_category for cat in filter_categories):
                    include_result = False
            
            # Brand filter
            if 'brands' in filters:
                result_brand = metadata.get('brand', '').lower()
                filter_brands = [brand.lower() for brand in filters['brands']]
                if result_brand not in filter_brands:
                    include_result = False
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def generate_description(self, metadata: Dict[str, Any]) -> str:
        """Generate AI description for a product"""
        if self.genai_model and self.genai_tokenizer:
            return self._generate_ai_description(metadata)
        else:
            return self._generate_template_description(metadata)
    
    def _generate_ai_description(self, metadata: Dict[str, Any]) -> str:
        """Generate description using AI model"""
        try:
            import torch
            
            title = metadata.get('title', '')
            category = metadata.get('category', '')
            material = metadata.get('material', '')
            color = metadata.get('color', '')
            
            prompt = f"Write a creative product description for: {{title}}"
            if category:
                prompt += f" Category: {{category}}"
            if material:
                prompt += f" Material: {{material}}"
            if color:
                prompt += f" Color: {{color}}"
            
            inputs = self.genai_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            with torch.no_grad():
                outputs = self.genai_model.generate(
                    inputs.input_ids,
                    max_length=80,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.genai_tokenizer.eos_token_id
                )
            
            description = self.genai_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return description.strip()
            
        except Exception as e:
            logger.warning(f"AI description generation failed: {{str(e)}}")
            return self._generate_template_description(metadata)
    
    def _generate_template_description(self, metadata: Dict[str, Any]) -> str:
        """Generate description using templates"""
        title = metadata.get('title', 'furniture piece')
        category = metadata.get('category', '').lower()
        material = metadata.get('material', '').lower()
        color = metadata.get('color', '').lower()
        
        templates = [
            f"Discover the perfect blend of style and functionality with this {{color}} {{category}}.",
            f"Transform your space with this beautifully crafted {{material}} {{category}}.",
            f"Experience comfort and elegance with this premium {{category}} piece.",
            f"Add sophistication to your home with this {{color}} {{category}}."
        ]
        
        if color and category:
            return templates[0]
        elif material and category:
            return templates[1]
        elif category:
            return templates[2]
        else:
            return "Enhance your living space with this thoughtfully designed furniture piece."

# Usage in main.py lifespan function:
'''
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(" Starting AI-Powered Furniture Platform...")
    
    try:
        # Initialize managers
        data_manager = DataManager(settings.DATA_PATH)
        ai_manager = AIModelManager(settings)
        
        # Load pre-trained models instead of training from scratch
        success = await ai_manager.load_pretrained_models()
        if not success:
            logger.error("Failed to load pre-trained models")
            raise RuntimeError("Model loading failed")
        
        # Store in app state
        app.state.data_manager = data_manager
        app.state.ai_manager = ai_manager
        
        logger.info("✅ All systems ready!")
        
    except Exception as e:
        logger.error(f" Startup failed: {{str(e)}}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
'''

# Model Configuration (from training artifacts):
MODEL_CONFIG = {self.config}

logger.info(" AI Models integration code generated successfully")
logger.info(" Next steps:")
logger.info("   1. Copy this code to backend/ai_models.py")
logger.info("   2. Update main.py with the new lifespan function")
logger.info("   3. Copy model artifacts to backend/models/")
logger.info("   4. Test the integration")
'''
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(integration_code)
            logger.info(f"💾 Integration code saved to {output_path}")
        
        return integration_code
    
    def copy_model_artifacts_to_backend(self, backend_models_dir: str) -> bool:
        """
        Copy model artifacts to backend directory
        
        Args:
            backend_models_dir: Path to backend models directory
            
        Returns:
            True if successful
        """
        try:
            backend_path = Path(backend_models_dir)
            backend_path.mkdir(exist_ok=True)
            
            artifacts = [
                "model_config.json",
                "product_embeddings.npy", 
                "vector_db.pkl",
                "product_metadata.pkl"
            ]
            
            for artifact in artifacts:
                src = self.models_dir / artifact
                dst = backend_path / artifact
                
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info(f"✅ Copied {artifact}")
                else:
                    logger.error(f"❌ Missing {artifact}")
                    return False
            
            logger.info(f"🎉 All model artifacts copied to {backend_models_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy artifacts: {str(e)}")
            return False
    
    def run_deployment_checklist(self) -> Dict[str, bool]:
        """
        Run complete deployment checklist
        
        Returns:
            Dictionary with checklist results
        """
        logger.info("🔍 Running deployment checklist...")
        
        checklist = {}
        
        # 1. Validate artifacts
        checklist['artifacts_valid'] = all(self.validate_model_artifacts().values())
        
        # 2. Load config
        try:
            self.load_model_config()
            checklist['config_loaded'] = True
        except:
            checklist['config_loaded'] = False
        
        # 3. Check model performance
        if self.config and 'performance_metrics' in self.config:
            metrics = self.config['performance_metrics']
            checklist['performance_acceptable'] = (
                metrics.get('precision', 0) > 0.3 and 
                metrics.get('avg_search_time', 1) < 0.5
            )
        else:
            checklist['performance_acceptable'] = False
        
        # 4. Check dependencies
        try:
            import sentence_transformers
            import sklearn
            import numpy
            checklist['dependencies_available'] = True
        except ImportError:
            checklist['dependencies_available'] = False
        
        # Summary
        all_passed = all(checklist.values())
        status = "✅ READY FOR DEPLOYMENT" if all_passed else "❌ ISSUES FOUND"
        
        logger.info(f"\n🎯 Deployment Checklist Results:")
        for check, result in checklist.items():
            status_icon = "✅" if result else "❌"
            logger.info(f"   {status_icon} {check.replace('_', ' ').title()}")
        
        logger.info(f"\n{status}")
        
        return checklist

def main():
    """Main deployment integration function"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize deployment manager
    deploy_manager = ModelDeploymentManager()
    
    # Run deployment checklist
    checklist = deploy_manager.run_deployment_checklist()
    
    if all(checklist.values()):
        # Generate integration code
        integration_code = deploy_manager.generate_backend_integration_code(
            "deployment_integration.py"
        )
        
        logger.info("🎉 Deployment integration complete!")
        logger.info("📋 Files generated:")
        logger.info("   - deployment_integration.py (Backend integration code)")
        
    else:
        logger.error("❌ Deployment checklist failed. Fix issues before deploying.")

if __name__ == "__main__":
    main()