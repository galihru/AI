#!/usr/bin/env python3
"""Collect all Arduino files into a single corpus file."""
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def collect_files(paths, extensions=['.ino', '.cpp', '.h', '.c']):
    """Recursively collect all files with specified extensions."""
    all_files = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path not found: {path}")
            continue
        
        if path.is_file():
            all_files.append(path)
        else:
            for ext in extensions:
                all_files.extend(path.rglob(f'*{ext}'))
    
    return all_files


def main():
    parser = argparse.ArgumentParser(description='Collect Arduino corpus')
    parser.add_argument('--corpus', required=True, help='Semicolon-separated paths')
    parser.add_argument('--out', required=True, help='Output corpus file')
    args = parser.parse_args()
    
    paths = args.corpus.split(';')
    print(f"Collecting from {len(paths)} path(s)...")
    
    files = collect_files(paths)
    print(f"Found {len(files)} files")
    
    total_lines = 0
    with open(args.out, 'w', encoding='utf-8', errors='ignore') as outf:
        for filepath in tqdm(files, desc="Processing"):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as inf:
                    content = inf.read()
                    outf.write(f"\n// FILE: {filepath}\n")
                    outf.write(content)
                    outf.write("\n")
                    total_lines += content.count('\n')
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    print(f"✓ Corpus saved: {args.out}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Total files: {len(files):,}")


if __name__ == '__main__':
    main()
