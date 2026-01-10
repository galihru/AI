#!/usr/bin/env python3
"""Build TF-IDF retrieval index from Arduino files."""
import argparse
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def collect_files(paths, extensions=['.ino', '.cpp', '.h', '.c']):
    """Collect all source files."""
    all_files = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        if path.is_file():
            all_files.append(path)
        else:
            for ext in extensions:
                all_files.extend(path.rglob(f'*{ext}'))
    return all_files


def main():
    parser = argparse.ArgumentParser(description='Build retrieval index')
    parser.add_argument('--corpus', required=True, help='Semicolon-separated paths')
    parser.add_argument('--out', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    paths = args.corpus.split(';')
    files = collect_files(paths)
    
    print(f"Building index from {len(files)} files...")
    
    documents = []
    file_paths = []
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                documents.append(content)
                file_paths.append(str(filepath))
        except:
            pass
    
    print("Computing TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Store index
    index_data = {
        'files': file_paths,
        'documents': documents[:500],  # Store first 500 for quick access
        'vocab': vectorizer.get_feature_names_out().tolist()
    }
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"✓ Index saved: {args.out}")
    print(f"  Indexed files: {len(file_paths):,}")
    print(f"  Vocab size: {len(index_data['vocab'])}")


if __name__ == '__main__':
    main()
