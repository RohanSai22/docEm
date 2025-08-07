# DocEm - Document Embedding and Monitoring System

🚀 A comprehensive real-time document monitoring and semantic search system that automatically processes files and enables intelligent search across text and image content.

## ✨ Features

- **🔄 Real-time File Monitoring**: Automatically detects new, modified, and deleted files
- **📄 Multi-format Support**: Processes TXT, MD, PDF, DOCX, CSV, XLSX, images (JPG, PNG, etc.)
- **🧠 Semantic Search**: Advanced similarity search with k=2 results using state-of-the-art embeddings
- **🖼️ Cross-modal Search**: Search images using text queries and vice versa
- **⚡ Auto-update**: Automatically updates embeddings when files are modified
- **💾 In-memory Database**: Fast ChromaDB storage for instant search results
- **🎯 Precise Results**: Returns top 2 most relevant results for focused search

## Sample Output :
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2148dbc2-868b-463b-905a-e9714fcc11e8" />


## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or download** the repository
2. **Install dependencies**:

   ```bash
   uv venv --python 3.11
   ```

   ```bash
   .venv\Scripts\activate
   ```

   ```bash
   uv pip install -r requirements.txt
   ```

3. **Run the system**:
   ```bash
   python main.py
   ```

## 📦 Dependencies

The system uses the following key libraries:

- **chromadb**: Vector database for embedding storage
- **sentence-transformers**: Text embedding generation
- **transformers**: CLIP model for image embeddings
- **torch**: PyTorch for ML model inference
- **watchdog**: File system monitoring
- **langchain**: Advanced text chunking
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **pandas**: Excel/CSV data processing
- **Pillow**: Image processing
- **openpyxl**: Excel file support

## 🚀 Quick Start

### 1. Basic Usage

```bash
python main.py
```

### 2. Setup Your Test Directory

When prompted, enter the path to your directory containing files to monitor:

```
📁 Enter directory path to monitor (or 'quit' to exit): test
```

**Tip**: You can use `test` or `./test` for a local test directory, or provide any absolute path.

### 3. File Upload

Create a `test` directory and add your files:

```bash
mkdir test
# Copy your files to the test directory
cp your_documents/* test/
cp your_images/* test/
```

### 4. Start Searching

Once the system is running, you can:

- **Search text**: `"machine learning algorithms"`
- **Search by image**: `path/to/your/image.jpg`
- **View statistics**: `stats`
- **Exit**: `quit` or `exit`

## 📁 Supported File Formats

### Text Documents

- **Plain Text**: `.txt`, `.md`, `.py`, `.js`, `.java`, `.cpp`, `.c`, `.html`, `.css`
- **Documents**: `.pdf`, `.docx`, `.doc`
- **Data Files**: `.csv`, `.xlsx`, `.xls`

### Images

- **Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`
- **Features**: CLIP-based embeddings for semantic image search

## 🔍 Search Examples

### Text Search

```
🔍 Enter query: python data analysis
```

### Image Search

```
🔍 Enter query: test/chart.png
```

### Cross-modal Search

```
🔍 Enter query: business chart visualization
# This will find both text mentioning charts AND relevant images

```

## 💡 System Architecture

```
📁 Watch Directory
    ↓
🔍 File Monitor (Watchdog)
    ↓
📄 Document Processor
    ↓
✂️ Text Chunker (LangChain)
    ↓
🧠 Embedding Manager
    ├── 📝 Text Model (BGE)
    └── 🖼️ Image Model (CLIP)
    ↓
💾 ChromaDB Storage
    ├── 📚 Text Collection
    └── 🎨 Image Collection
    ↓
🔍 Semantic Search (k=2)
```

## ⚙️ Configuration

### Embedding Models

The system uses these pre-configured models:

- **Text**: `BAAI/bge-small-en-v1.5` (High-quality, efficient)
- **Images**: `openai/clip-vit-base-patch32` (Cross-modal capabilities)

### Search Parameters

- **Results returned**: k=2 (top 2 most relevant)
- **Chunk size**: 500 characters with 100 character overlap
- **Update frequency**: Real-time on file changes

## 🔧 Advanced Usage

### Custom Directory Monitoring

```python
from main import DocEmSystem

system = DocEmSystem()
system.start_monitoring("/path/to/your/documents")

# Search programmatically
results = system.search("your query", k=2)
```

### Batch Processing

```python
# The system automatically processes all files in the directory
# No manual batch processing needed - just add files to the watched folder!
```

## 📊 System Statistics

Use the `stats` command to view:

- Number of text chunks indexed
- Number of images processed
- Total files being tracked
- Monitoring status
- Current watch directory

Example output:

```
📊 SYSTEM STATISTICS:
  Text chunks: 156
  Images: 23
  Files tracked: 45
  Monitoring: True
  Directory: /path/to/your/test
```

## 🔄 Real-time Updates

The system automatically:

1. **Detects new files** → Processes and indexes immediately
2. **Detects file changes** → Updates embeddings automatically
3. **Detects deleted files** → Removes from index
4. **Handles file moves** → Updates file paths

## 🚨 Troubleshooting

### Common Issues

1. **"No results found"**

   - Check if files were processed successfully
   - Try broader search terms
   - Use `stats` to verify file count

2. **Model loading errors**

   - Ensure stable internet connection for first run
   - Models are downloaded automatically (~1-2GB total)

3. **File processing errors**
   - Check file permissions
   - Verify file formats are supported
   - Check logs in `docem.log`

### Performance Tips

- **Large files**: System chunks text automatically for optimal performance
- **Many files**: Processing happens in background, search remains responsive
- **Memory usage**: In-memory database is fast but uses RAM

## 📝 Logging

The system creates detailed logs in `docem.log` including:

- File processing status
- Embedding generation results
- Search queries and results
- Error messages and debugging info

## 🤝 Contributing

Feel free to:

- Report issues
- Suggest new file formats
- Improve embedding models
- Add new features

## 📄 License

Open source - feel free to use and modify!

## 🔮 Future Enhancements

- [ ] Persistent database option
- [ ] Web interface
- [ ] More embedding models
- [ ] OCR for scanned documents
- [ ] Audio file support
- [ ] API endpoints

---

**Happy Searching! 🎉**

For support or questions, check the logs or create an issue in the repository.

