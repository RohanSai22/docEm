# Quick Start Guide for DocEm

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
uv venv --python 3.11
```

```bash
venv\Scripts\activat
```

````bash
uv pip install -r requirements.txt
```

### 2. Create Test Directory

```bash
mkdir test
````

### 3. Add Your Files

Copy your documents and images to the `test` directory:

```bash
# Example files you can add:
# - PDF documents
# - Word documents (.docx)
# - Text files (.txt, .md)
# - CSV/Excel files
# - Images (.jpg, .png)
# - Code files (.py, .js, etc.)
```

### 4. Run the System

```bash
python main_new.py
```

### 5. Monitor Your Directory

When prompted, enter `test` (or the full path to your directory):

```
📁 Enter directory path to monitor: test
```

## 🔍 How to Search

### Text Search Examples:

```
🔍 Enter query: machine learning
🔍 Enter query: financial analysis
🔍 Enter query: python programming
```

### Image Search Examples:

```
🔍 Enter query: test/chart.png
🔍 Enter query: business visualization
```

### System Commands:

```
🔍 Enter query: stats    # View system statistics
🔍 Enter query: quit     # Exit the program
```

## ✅ Expected Behavior

1. **Initial Processing**: System will process all files in your directory
2. **Real-time Monitoring**: Any new files added will be automatically processed
3. **File Updates**: Modified files will have their embeddings updated
4. **Search Results**: Returns top 2 most relevant results with similarity scores

## 📊 Example Output

```
📊 SYSTEM STATISTICS:
  Text chunks: 156
  Images: 23
  Files tracked: 45
  Monitoring: True
  Directory: /path/to/test

📋 SEARCH RESULTS (Top 2):
--------------------------------------------------

📄 TEXT RESULTS:

  Result 1:
    📁 File: document.pdf
    📊 Similarity: 0.8234
    📑 Chunk: 2/5
    📝 Content: This document discusses machine learning algorithms...

🖼️  IMAGE RESULTS:

  Result 1:
    📁 File: chart.png
    📊 Similarity: 0.7891
    🎨 Type: .png
```

## 🚨 Troubleshooting

- **No results**: Check that files were processed (use `stats` command)
- **Slow initial startup**: First run downloads ML models (~1-2GB)
- **Permission errors**: Ensure read/write access to the directory
- **Memory issues**: System uses in-memory database (restart if needed)

## 💡 Tips

- Use specific search terms for better results
- Mix text and image searches for comprehensive results
- System works best with diverse file types
- Check `docem.log` for detailed processing information

```

```
