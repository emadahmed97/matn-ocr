# Arabic Manuscript Dataset

## Directory Structure
```
data/manuscripts/
├── images/           # Place your manuscript images here
├── transcriptions/   # Place corresponding text files here
├── dataset.csv       # CSV with image paths and texts (optional)
└── metadata.json     # JSON metadata file (optional)
```

## Supported Formats

### Images
- JPG, PNG, TIFF formats
- 300+ DPI recommended
- Sequential naming (e.g., page_001.jpg)

### Text Files
- UTF-8 encoded .txt files
- Same name as image (e.g., page_001.txt)
- One transcription per file

### CSV Format
```csv
image_path,text,split,source,page_number,book_title
images/page_001.jpg,"بسم الله الرحمن الرحيم",train,manuscript,1,"كتاب الطهارة"
```
