# DocEm - Document Embedding and Monitoring System

 Real-time document monitoring and semantic search system with automatic file processing and intelligent cross-modal search.

##  Key Features

- **🔄 Real-time Monitoring**: Auto-detects new, modified, and deleted files
- **📄 Multi-format Support**: TXT, MD, PDF, DOCX, CSV, XLSX, images (JPG, PNG, etc.)
- **🧠 Semantic Search**: k=2 results using BGE text embeddings and CLIP image embeddings
- **🖼️ Cross-modal Search**: Find images with text queries and vice versa.But there is a limitation to it due to data size, as size increases it will rank better...
- **⚡ Auto-update**: Embeddings update automatically when files change

## Sample Output
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2148dbc2-868b-463b-905a-e9714fcc11e8" />

## Quick Start

```bash
# Setup environment
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt

# Create test directory and add your files
mkdir test
# Copy your documents and images to test/

# Run the system
python main.py
```

When prompted, enter `test` as your directory to monitor.

##  Usage Examples

```bash
🔍 Enter query: machine learning        # Text search
🔍 Enter query: test/chart.png         # Image search  
🔍 Enter query: business visualization  # Cross-modal search
🔍 Enter query: stats                  # System statistics
🔍 Enter query: quit                   # Exit
```

## System Architecture

```
📁 Watch Directory → 🔍 File Monitor → 📄 Document Processor 
    ↓
✂️ Text Chunker → 🧠 Embeddings (BGE + CLIP) → 💾 ChromaDB
    ↓
🔍 Semantic Search (k=2 results)
```

##  Supported Formats

**Text**: `.txt`, `.md`, `.py`, `.pdf`, `.docx`, `.csv`, `.xlsx`, code files  
**Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`

## Troubleshooting

- **No results**: Use `stats` command to check file processing
- **Slow startup**: First run downloads ML models (~1-2GB)
- **Issues**: Check `docem.log` for detailed error information

##  Key Dependencies

`chromadb`, `sentence-transformers`, `transformers`, `torch`, `watchdog`, `langchain`, `PyPDF2`, `python-docx`, `pandas`, `Pillow`

---

**Happy Searching** Check logs in `docem.log` for detailed system information.
