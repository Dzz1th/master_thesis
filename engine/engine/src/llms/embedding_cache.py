"""
Embedding cache management for sentiment analysis pipeline
"""
import os
import pickle
import hashlib
import logging
import json
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

class EmbeddingCache:
    """Cache manager for text embeddings that stores texts alongside embeddings"""
    
    def __init__(self, cache_dir: str):
        """Initialize embedding cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_path(self, name: str) -> str:
        """Get path for cache file
        
        Args:
            name: Cache name identifier
            
        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{name}_cache.pkl")
    
    def get_manifest_path(self, name: str) -> str:
        """Get path for cache manifest file
        
        Args:
            name: Cache name identifier
            
        Returns:
            Path to manifest file
        """
        return os.path.join(self.cache_dir, f"{name}_manifest.json")
    
    def compute_text_hash(self, text: str) -> str:
        """Compute hash for text
        
        Args:
            text: Text to hash
            
        Returns:
            Text hash
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def compute_texts_hash(self, texts: List[str]) -> str:
        """Compute hash for list of texts
        
        Args:
            texts: List of texts to hash
            
        Returns:
            Combined hash for all texts
        """
        # Sort texts to ensure consistent hash
        sorted_texts = sorted(texts)
        
        # Combine hashes of individual texts
        hashes = [self.compute_text_hash(text) for text in sorted_texts]
        combined_hash = hashlib.md5(''.join(hashes).encode('utf-8')).hexdigest()
        
        return combined_hash
    
    def create_manifest(self, name: str, texts: List[str]) -> Dict[str, Any]:
        """Create cache manifest for texts
        
        Args:
            name: Cache name identifier
            texts: List of texts
            
        Returns:
            Manifest dictionary
        """
        # Compute hash for entire text corpus
        corpus_hash = self.compute_texts_hash(texts)
        
        # Create manifest
        manifest = {
            'name': name,
            'corpus_hash': corpus_hash,
            'count': len(texts),
            'text_hashes': {
                i: self.compute_text_hash(text) for i, text in enumerate(texts)
            }
        }
        
        return manifest
    
    def save_manifest(self, name: str, manifest: Dict[str, Any]):
        """Save manifest to disk
        
        Args:
            name: Cache name identifier
            manifest: Manifest dictionary
        """
        manifest_path = self.get_manifest_path(name)
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.debug(f"Saved manifest to {manifest_path}")
    
    def load_manifest(self, name: str) -> Optional[Dict[str, Any]]:
        """Load manifest from disk
        
        Args:
            name: Cache name identifier
            
        Returns:
            Manifest dictionary or None if not found
        """
        manifest_path = self.get_manifest_path(name)
        
        if not os.path.exists(manifest_path):
            return None
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            return manifest
        except Exception as e:
            self.logger.warning(f"Error loading manifest: {e}")
            return None
    
    def save_embeddings(self, name: str, texts: List[str], embeddings: np.ndarray):
        """Save embeddings and texts to cache
        
        Args:
            name: Cache name identifier
            texts: List of texts
            embeddings: Array of embeddings
        """
        cache_path = self.get_cache_path(name)
        
        # Create manifest
        manifest = self.create_manifest(name, texts)
        
        # Save embeddings and manifest
        cache_data = {
            'texts': texts,
            'embeddings': embeddings,
            'manifest': manifest
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Save separate manifest file for quick validation
        self.save_manifest(name, manifest)
        
        self.logger.info(f"Saved {len(texts)} embeddings to {cache_path}")
    
    def load_embeddings(self, name: str, texts: List[str]) -> Optional[np.ndarray]:
        """Load embeddings from cache if they exist and are valid
        
        Args:
            name: Cache name identifier
            texts: List of texts to validate against cache
            
        Returns:
            Array of embeddings or None if cache is invalid or missing
        """
        cache_path = self.get_cache_path(name)
        
        if not os.path.exists(cache_path):
            self.logger.info(f"No cache found for {name}")
            return None
        
        # Try to load cache
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is valid
            if self._validate_cache(cache_data, texts):
                self.logger.info(f"Loaded {len(cache_data['embeddings'])} valid embeddings from {cache_path}")
                return cache_data['embeddings']
            else:
                self.logger.warning(f"Cache {name} is invalid, will recompute embeddings")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error loading cache: {e}")
            return None
    
    def _validate_cache(self, cache_data: Dict[str, Any], texts: List[str]) -> bool:
        """Validate cache against current texts
        
        Args:
            cache_data: Cache data dictionary
            texts: List of texts to validate against
            
        Returns:
            True if cache is valid, False otherwise
        """
        # Check if cache contains expected fields
        if 'texts' not in cache_data or 'embeddings' not in cache_data or 'manifest' not in cache_data:
            self.logger.warning("Cache missing required fields")
            return False
        
        # Check if number of texts matches
        if len(cache_data['texts']) != len(texts):
            self.logger.warning(f"Cache text count mismatch: {len(cache_data['texts'])} vs {len(texts)}")
            return False
        
        # Compare text content
        # First compute hash of current texts
        current_corpus_hash = self.compute_texts_hash(texts)
        cached_corpus_hash = cache_data['manifest']['corpus_hash']
        
        if current_corpus_hash != cached_corpus_hash:
            # Texts have changed
            self.logger.warning(f"Cache corpus hash mismatch: changes detected in texts")
            
            # For debugging, find which texts changed
            current_hashes = {i: self.compute_text_hash(text) for i, text in enumerate(texts)}
            cached_hashes = cache_data['manifest']['text_hashes']
            
            different_indices = []
            for i in current_hashes:
                if str(i) in cached_hashes and current_hashes[i] != cached_hashes[str(i)]:
                    different_indices.append(i)
            
            if different_indices:
                self.logger.debug(f"Changed texts at indices: {different_indices[:5]}{'...' if len(different_indices) > 5 else ''}")
            
            return False
        
        # Check if embeddings shape matches text count
        if len(cache_data['embeddings']) != len(texts):
            self.logger.warning(f"Cache embeddings count mismatch: {len(cache_data['embeddings'])} vs {len(texts)}")
            return False
        
        return True
    
    def clear_cache(self, name: Optional[str] = None):
        """Clear cache files
        
        Args:
            name: Cache name to clear, or None to clear all caches
        """
        if name:
            cache_path = self.get_cache_path(name)
            manifest_path = self.get_manifest_path(name)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                self.logger.info(f"Cleared cache file: {cache_path}")
            
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
                self.logger.info(f"Cleared manifest file: {manifest_path}")
        else:
            import glob
            
            # Find all cache files
            cache_files = glob.glob(os.path.join(self.cache_dir, "*_cache.pkl"))
            manifest_files = glob.glob(os.path.join(self.cache_dir, "*_manifest.json"))
            
            # Remove cache files
            for file_path in cache_files:
                os.remove(file_path)
                self.logger.info(f"Cleared cache file: {file_path}")
            
            # Remove manifest files
            for file_path in manifest_files:
                os.remove(file_path)
                self.logger.info(f"Cleared manifest file: {file_path}")