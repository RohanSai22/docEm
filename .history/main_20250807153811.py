#!/usr/bin/env python3
"""
DocEm - Document Embedding and Monitoring System
A comprehensive file monitoring and semantic search system that:
1. Monitors a directory for file changes
2. Extracts text from various file formats
3. Creates embeddings using sentence transformers and CLIP
4. Stores embeddings in ChromaDB (in-memory)
5. Provides interactive search interface with k=2 results
6. Automatically processes new files and updates modified files
"""

import os
import sys
import time
import hashlib
import uuid
import traceback
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import threading

# File monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Document processing
import PyPDF2
import docx
import pandas as pd
from PIL import Image

# Machine Learning models
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# Vector database
import chromadb

# Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docem.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles extraction of text content from various file formats"""
    
    def __init__(self):
        self.supported_text_extensions = {
            '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.html', '.css',
            '.csv', '.xlsx', '.xls', '.pdf', '.docx', '.doc'
        }
        self.supported_image_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'
        }
        self.supported_extensions = self.supported_text_extensions | self.supported_image_extensions
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported"""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if file is an image"""
        return Path(file_path).suffix.lower() in self.supported_image_extensions
    
    def process_document(self, file_path: str) -> Optional[str]:
        """
        Processes a document file and extracts its text content based on file type.
        
        Args:
            file_path (str): The path to the document file.
            
        Returns:
            str: The extracted raw text content, or None if an error occurs.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if extension in ['.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.html', '.css']:
                return self._extract_text_file(file_path)
            elif extension == '.csv':
                return self._extract_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                return self._extract_excel(file_path)
            elif extension == '.pdf':
                return self._extract_pdf(file_path)
            elif extension in ['.docx', '.doc']:
                return self._extract_docx(file_path)
            else:
                logger.warning(f"Unsupported file type for text extraction: {extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text files"""
        encoding = self._detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _extract_csv(self, file_path: str) -> str:
        """Extract text from CSV files"""
        try:
            df = pd.read_csv(file_path)
            text_content = []
            text_content.append(f"CSV file: {Path(file_path).name}")
            text_content.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_content.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Add sample data (first few rows)
            for idx, row in df.head(10).iterrows():
                row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
                text_content.append(f"Row {idx}: {row_text}")
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {str(e)}")
            return f"CSV file: {Path(file_path).name} (Error reading content)"
    
    def _extract_excel(self, file_path: str) -> str:
        """Extract text from Excel files"""
        try:
            df = pd.read_excel(file_path)
            text_content = []
            text_content.append(f"Excel file: {Path(file_path).name}")
            text_content.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_content.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Add sample data
            for idx, row in df.head(10).iterrows():
                row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
                text_content.append(f"Row {idx}: {row_text}")
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error reading Excel {file_path}: {str(e)}")
            return f"Excel file: {Path(file_path).name} (Error reading content)"
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text_content = []
            text_content.append(f"PDF file: {Path(file_path).name}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"Page {page_num + 1}: {page_text}")
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return f"PDF file: {Path(file_path).name} (Error reading content)"
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(file_path)
            text_content = []
            text_content.append(f"DOCX file: {Path(file_path).name}")
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return f"DOCX file: {Path(file_path).name} (Error reading content)"

class TextChunker:
    """Handles text chunking using LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 500, overlap_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap_size: int = None) -> List[str]:
        """
        Splits the input text into overlapping chunks using LangChain.
        
        Args:
            text (str): The raw text content to chunk.
            chunk_size (int): Override default chunk size.
            overlap_size (int): Override default overlap size.
            
        Returns:
            list: A list of text chunks.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Use custom parameters if provided
        if chunk_size or overlap_size:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or self.chunk_size,
                chunk_overlap=overlap_size or self.overlap_size,
                length_function=len,
                is_separator_regex=False,
            )
            return splitter.split_text(text)
        
        return self.text_splitter.split_text(text)

class EmbeddingManager:
    """Manages text and image embeddings with separate models"""
    
    def __init__(self):
        self.text_model = None
        self.image_model = None
        self.image_processor = None
        self.text_model_name = "BAAI/bge-small-en-v1.5"
        self.image_model_name = "openai/clip-vit-base-patch32"
        logger.info("EmbeddingManager initialized")
    
    def load_text_embedding_model(self, model_name: str = None):
        """Load text embedding model using SentenceTransformer"""
        model_name = model_name or self.text_model_name
        logger.info(f"Loading text embedding model: {model_name}")
        try:
            self.text_model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded text model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading text model {model_name}: {e}")
            self.text_model = None
    
    def load_image_embedding_model(self, model_name: str = None):
        """Load image embedding model and processor using CLIP"""
        model_name = model_name or self.image_model_name
        logger.info(f"Loading image embedding model: {model_name}")
        try:
            self.image_model = CLIPModel.from_pretrained(model_name)
            self.image_processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"Successfully loaded image model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading image model {model_name}: {e}")
            self.image_model = None
            self.image_processor = None
    
    def get_text_embedding(self, text_chunks: List[str]) -> Optional[List[List[float]]]:
        """Generate normalized embeddings for text chunks"""
        if self.text_model is None:
            logger.error("Text embedding model is not loaded")
            return None
        
        if not isinstance(text_chunks, list) or not text_chunks:
            return []
        
        try:
            embeddings = self.text_model.encode(text_chunks, convert_to_tensor=True, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            return None
    
    def get_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """Generate normalized embedding for an image file"""
        if self.image_model is None or self.image_processor is None:
            logger.error("Image embedding model or processor is not loaded")
            return None
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.image_model.get_image_features(pixel_values=inputs.pixel_values)
                # Normalize the features for proper cosine similarity
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            
            return image_features.tolist()[0]
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def get_text_features_for_image_search(self, text_query: str) -> Optional[List[float]]:
        """Generate normalized text features using CLIP for cross-modal search"""
        if self.image_model is None or self.image_processor is None:
            logger.error("Image model not loaded for cross-modal search")
            return None
        
        try:
            text_inputs = self.image_processor(text=[text_query], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.image_model.get_text_features(text_inputs.input_ids)
                # Normalize the features for proper cosine similarity
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            return text_features.tolist()[0]
        except Exception as e:
            logger.error(f"Error generating text features for image search: {e}")
            return None

class FileMonitor(FileSystemEventHandler):
    """Monitors file system changes and triggers processing"""
    
    def __init__(self, callback):
        self.callback = callback
        super().__init__()
    
    def on_modified(self, event):
        if not event.is_directory:
            self.callback('modified', event.src_path)
    
    def on_created(self, event):
        if not event.is_directory:
            self.callback('created', event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.callback('deleted', event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory:
            self.callback('deleted', event.src_path)
            self.callback('created', event.dest_path)

class DocEmSystem:
    """Main document embedding and monitoring system"""
    
    def __init__(self):
        self.watch_directory = None
        self.chroma_client = None
        self.text_collection = None
        self.image_collection = None
        self.doc_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.observer = None
        self.monitoring = False
        self.text_collection_name = "docem_text_embeddings"
        self.image_collection_name = "docem_image_embeddings"
        self.file_hashes = {}  # Track file hashes for change detection
        
        # Initialize components
        self._initialize_models()
        self._initialize_chroma()
    
    def _cosine_distance_to_similarity(self, distance: float) -> float:
        """
        Convert cosine distance to similarity score.
        
        With cosine distance metric and normalized vectors:
        - Cosine distance = 1 - cosine_similarity
        - Therefore: cosine_similarity = 1 - cosine_distance
        
        Distance ranges: 0 (identical) to 2 (opposite)
        Similarity ranges: 1.0 (identical) to -1.0 (opposite)
        
        Args:
            distance (float): Cosine distance from ChromaDB
            
        Returns:
            float: Similarity score between -1.0 and 1.0
        """
        # Clamp distance to valid range [0, 2] to handle any numerical precision issues
        distance = max(0.0, min(2.0, distance))
        return 1.0 - distance
    
    def _initialize_models(self):
        """Initialize embedding models"""
        logger.info("Initializing embedding models...")
        self.embedding_manager.load_text_embedding_model()
        self.embedding_manager.load_image_embedding_model()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collections"""
        try:
            logger.info("Initializing in-memory ChromaDB...")
            self.chroma_client = chromadb.Client()
            
            # Create collections with cosine similarity for proper bounded similarity scores
            self.text_collection = self.chroma_client.get_or_create_collection(
                name=self.text_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.image_collection = self.chroma_client.get_or_create_collection(
                name=self.image_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB collections created successfully with cosine similarity")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content to detect changes"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return ""
    
    def _file_change_callback(self, event_type: str, file_path: str):
        """Handle file system events"""
        logger.info(f"File {event_type}: {file_path}")
        
        if event_type == 'deleted':
            self._remove_file_embeddings(file_path)
        else:
            # For created and modified events
            if self.doc_processor.is_supported(file_path):
                # Add small delay to ensure file is fully written
                time.sleep(0.5)
                self._process_file(file_path)
    
    def _remove_file_embeddings(self, file_path: str):
        """Remove embeddings for a deleted file from both collections"""
        try:
            # Remove from text collection
            text_results = self.text_collection.get(where={"file_path": file_path})
            if text_results['ids']:
                self.text_collection.delete(ids=text_results['ids'])
                logger.info(f"Removed {len(text_results['ids'])} text embeddings for {file_path}")
            
            # Remove from image collection
            image_results = self.image_collection.get(where={"file_path": file_path})
            if image_results['ids']:
                self.image_collection.delete(ids=image_results['ids'])
                logger.info(f"Removed {len(image_results['ids'])} image embeddings for {file_path}")
            
            # Remove from file hash tracking
            if file_path in self.file_hashes:
                del self.file_hashes[file_path]
                
        except Exception as e:
            logger.error(f"Error removing embeddings for {file_path}: {str(e)}")
    
    def _process_file(self, file_path: str):
        """Process a single file and create embeddings"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists: {file_path}")
                return
            
            # Check if file has changed
            current_hash = self._get_file_hash(file_path)
            if file_path in self.file_hashes and self.file_hashes[file_path] == current_hash:
                logger.debug(f"File unchanged, skipping: {file_path}")
                return
            
            # Remove old embeddings if file exists
            if file_path in self.file_hashes:
                self._remove_file_embeddings(file_path)
            
            logger.info(f"Processing file: {file_path}")
            
            if self.doc_processor.is_image_file(file_path):
                self._process_image_file(file_path, current_hash)
            else:
                self._process_text_file(file_path, current_hash)
            
            # Update file hash
            self.file_hashes[file_path] = current_hash
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def _process_text_file(self, file_path: str, file_hash: str):
        """Process text file and create embeddings"""
        try:
            # Extract text
            text_content = self.doc_processor.process_document(file_path)
            if not text_content:
                logger.warning(f"No text extracted from {file_path}")
                return
            
            # Chunk text
            chunks = self.text_chunker.chunk_text(text_content)
            if not chunks:
                logger.warning(f"No chunks generated from {file_path}")
                return
            
            # Generate embeddings
            embeddings = self.embedding_manager.get_text_embedding(chunks)
            if not embeddings:
                logger.error(f"Failed to generate embeddings for {file_path}")
                return
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    'file_path': file_path,
                    'chunk_index': i,
                    'file_type': Path(file_path).suffix.lower(),
                    'file_hash': file_hash,
                    'total_chunks': len(chunks)
                })
                documents.append(chunk)
            
            # Add to ChromaDB
            self.text_collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} text chunks for {Path(file_path).name}")
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
    
    def _process_image_file(self, file_path: str, file_hash: str):
        """Process image file and create embedding"""
        try:
            # Generate image embedding
            embedding = self.embedding_manager.get_image_embedding(file_path)
            if not embedding:
                logger.error(f"Failed to generate image embedding for {file_path}")
                return
            
            # Prepare data for ChromaDB
            image_id = str(uuid.uuid4())
            metadata = {
                'file_path': file_path,
                'file_type': Path(file_path).suffix.lower(),
                'file_hash': file_hash
            }
            
            # Add to ChromaDB
            self.image_collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[image_id]
            )
            
            logger.info(f"Added image embedding for {Path(file_path).name}")
            
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
    
    def start_monitoring(self, directory: str):
        """Start monitoring a directory"""
        self.watch_directory = os.path.abspath(directory)
        
        if not os.path.exists(self.watch_directory):
            raise ValueError(f"Directory does not exist: {self.watch_directory}")
        
        logger.info(f"Starting to monitor directory: {self.watch_directory}")
        
        # Process existing files
        self._scan_directory()
        
        # Start file system monitoring
        self.observer = Observer()
        event_handler = FileMonitor(self._file_change_callback)
        self.observer.schedule(event_handler, self.watch_directory, recursive=True)
        self.observer.start()
        self.monitoring = True
        
        logger.info("File monitoring started successfully!")
    
    def _scan_directory(self):
        """Scan directory and process all supported files"""
        logger.info("Scanning directory for existing files...")
        
        file_count = 0
        for root, dirs, files in os.walk(self.watch_directory):
            for file in files:
                file_path = os.path.join(root, file)
                if self.doc_processor.is_supported(file_path):
                    self._process_file(file_path)
                    file_count += 1
        
        logger.info(f"Processed {file_count} files from directory scan")
    
    def search(self, query: str, k: int = 2) -> Dict[str, Any]:
        """
        Search for documents using semantic similarity with k=2 results.
        
        Args:
            query (str): Search query (text string or path to image file).
            k (int): Number of results to return (default: 2).
            
        Returns:
            dict: Search results from text and image collections.
        """
        try:
            results = {'text_results': None, 'image_results': None}
            
            # Check collection stats
            text_count = self.text_collection.count()
            image_count = self.image_collection.count()
            logger.info(f"Collection stats - Text chunks: {text_count}, Images: {image_count}")
            
            # Determine if query is an image file path
            is_image_query = (isinstance(query, str) and 
                            os.path.exists(query) and 
                            Path(query).suffix.lower() in self.doc_processor.supported_image_extensions)
            
            if is_image_query:
                logger.info(f"Processing image query: {query}")
                # Generate image embedding
                query_embedding = self.embedding_manager.get_image_embedding(query)
                if query_embedding:
                    # Search in image collection
                    if image_count > 0:
                        results['image_results'] = self.image_collection.query(
                            query_embeddings=[query_embedding],
                            n_results=min(k, image_count),
                            include=['metadatas', 'distances']
                        )
                        logger.info(f"Found {len(results['image_results']['ids'][0])} image results")
            else:
                logger.info(f"Processing text query: '{query}'")
                # Generate text embedding
                query_embeddings = self.embedding_manager.get_text_embedding([query])
                if query_embeddings:
                    # Search in text collection
                    if text_count > 0:
                        results['text_results'] = self.text_collection.query(
                            query_embeddings=query_embeddings,
                            n_results=min(k, text_count),
                            include=['metadatas', 'documents', 'distances']
                        )
                        logger.info(f"Found {len(results['text_results']['ids'][0])} text results")
                    
                    # Cross-modal search (Text to Image) with CLIP
                    if (image_count > 0 and 
                        self.embedding_manager.image_model is not None):
                        text_features = self.embedding_manager.get_text_features_for_image_search(query)
                        if text_features:
                            results['image_results'] = self.image_collection.query(
                                query_embeddings=[text_features],
                                n_results=min(k, image_count),
                                include=['metadatas', 'distances']
                            )
                            logger.info(f"Found {len(results['image_results']['ids'][0])} image results from text query")
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {'text_results': None, 'image_results': None}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            text_count = self.text_collection.count()
            image_count = self.image_collection.count()
            
            return {
                'text_chunks': text_count,
                'images': image_count,
                'total_files_tracked': len(self.file_hashes),
                'monitoring': self.monitoring,
                'watch_directory': self.watch_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def stop_monitoring(self):
        """Stop file monitoring"""
        if self.observer and self.monitoring:
            self.observer.stop()
            self.observer.join()
            self.monitoring = False
            logger.info("File monitoring stopped")

def interactive_search_loop(system: DocEmSystem):
    """Interactive search loop with k=2 results"""
    print("\n" + "=" * 60)
    print("🔍 INTERACTIVE SEARCH MODE")
    print("=" * 60)
    print("Enter your search queries below.")
    print("Commands:")
    print("  - 'stats': Show system statistics")
    print("  - 'quit' or 'exit': Exit the program")
    print("  - Any text: Search for documents")
    print("  - Image file path: Search by image")
    print("=" * 60)
    
    try:
        while True:
            query = input("\n🔍 Enter query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            
            if query.lower() == 'stats':
                stats = system.get_stats()
                print("\n📊 SYSTEM STATISTICS:")
                print(f"  Text chunks: {stats.get('text_chunks', 0)}")
                print(f"  Images: {stats.get('images', 0)}")
                print(f"  Files tracked: {stats.get('total_files_tracked', 0)}")
                print(f"  Monitoring: {stats.get('monitoring', False)}")
                print(f"  Directory: {stats.get('watch_directory', 'None')}")
                continue
            
            if not query:
                print("Please enter a search query.")
                continue
            
            # Perform search with k=2
            print(f"\n🔍 Searching for: '{query}'...")
            results = system.search(query, k=2)
            
            # Display results
            print("\n📋 SEARCH RESULTS (Top 2):")
            print("-" * 50)
            
            # Text results
            text_results = results.get('text_results')
            if text_results and text_results.get('ids') and text_results['ids'][0]:
                print("\n📄 TEXT RESULTS:")
                for i in range(len(text_results['ids'][0])):
                    distance = text_results['distances'][0][i]
                    metadata = text_results['metadatas'][0][i]
                    document = text_results['documents'][0][i]
                    
                    # Convert distance to similarity using proper method
                    similarity = system._cosine_distance_to_similarity(distance)
                    
                    print(f"\n  Result {i+1}:")
                    print(f"    📁 File: {Path(metadata['file_path']).name}")
                    print(f"    📊 Similarity: {similarity:.4f}")
                    print(f"    📑 Chunk: {metadata['chunk_index']+1}/{metadata['total_chunks']}")
                    print(f"    📝 Content: {document[:200]}{'...' if len(document) > 200 else ''}")
            
            # Image results
            image_results = results.get('image_results')
            if image_results and image_results.get('ids') and image_results['ids'][0]:
                print("\n🖼️  IMAGE RESULTS:")
                for i in range(len(image_results['ids'][0])):
                    distance = image_results['distances'][0][i]
                    metadata = image_results['metadatas'][0][i]
                    
                    # Convert distance to similarity using proper method
                    similarity = system._cosine_distance_to_similarity(distance)
                    
                    print(f"\n  Result {i+1}:")
                    print(f"    📁 File: {Path(metadata['file_path']).name}")
                    print(f"    📊 Similarity: {similarity:.4f}")
                    print(f"    🎨 Type: {metadata['file_type']}")
            
            if (not text_results or not text_results.get('ids') or not text_results['ids'][0]) and \
               (not image_results or not image_results.get('ids') or not image_results['ids'][0]):
                print("❌ No results found.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Search interrupted by user.")

def main():
    """Main function to run the DocEm system"""
    print("=" * 60)
    print("🚀 DocEm - Document Embedding and Monitoring System")
    print("=" * 60)
    print("Features:")
    print("  ✅ Real-time file monitoring")
    print("  ✅ Text and image embedding")
    print("  ✅ Semantic search with k=2 results")
    print("  ✅ Auto-update on file changes")
    print("  ✅ Cross-modal search (text ↔ image)")
    print("=" * 60)
    
    # Initialize system
    try:
        print("\n🔧 Initializing system...")
        system = DocEmSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        sys.exit(1)
    
    # Get directory to monitor
    while True:
        directory = input("\n📁 Enter directory path to monitor (or 'quit' to exit): ").strip()
        
        if directory.lower() == 'quit':
            sys.exit(0)
        
        if not directory:
            print("❌ Please enter a valid directory path.")
            continue
        
        # Convert to absolute path and handle test directory case
        if directory == 'test' or directory == './test':
            directory = os.path.join(os.getcwd(), 'test')
            
        directory = os.path.abspath(directory)
        
        try:
            system.start_monitoring(directory)
            break
        except Exception as e:
            print(f"❌ Error starting monitoring: {str(e)}")
            print("Please try a different directory.")
    
    print(f"\n✅ Monitoring started for: {directory}")
    print("🔄 System is now watching for file changes...")
    print("📊 Processing existing files...")
    
    # Wait a moment for initial processing
    time.sleep(2)
    
    # Show initial stats
    stats = system.get_stats()
    print(f"\n📈 Initial stats: {stats['text_chunks']} text chunks, {stats['images']} images")
    
    # Start interactive search
    try:
        interactive_search_loop(system)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping system...")
    finally:
        system.stop_monitoring()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()