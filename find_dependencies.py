#!/usr/bin/env python3
"""
Comprehensive dependency finder for Arabic OCR training pipeline.
This script analyzes all Python files to find required dependencies.
"""

import os
import ast
import sys
import re
from pathlib import Path
from collections import defaultdict

def extract_imports_from_file(file_path):
    """Extract all import statements from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST to get imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except SyntaxError:
            # If AST parsing fails, use regex fallback
            pass

        # Regex fallback for imports
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
        ]

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                continue
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    imports.add(match.group(1))

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return imports

def find_all_dependencies(root_dir):
    """Find all dependencies in Python files."""
    all_imports = set()
    file_imports = defaultdict(set)

    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files")

    # Extract imports from each file
    for file_path in python_files:
        imports = extract_imports_from_file(file_path)
        file_imports[file_path] = imports
        all_imports.update(imports)

    return all_imports, file_imports

def classify_imports(imports):
    """Classify imports into standard library vs external packages."""
    # Common standard library modules
    stdlib_modules = {
        'os', 'sys', 'json', 'ast', 're', 'io', 'math', 'logging', 'tempfile',
        'pathlib', 'datetime', 'typing', 'dataclasses', 'collections',
        'importlib', 'functools', 'itertools', 'copy', 'pickle', 'base64',
        'hashlib', 'uuid', 'time', 'random', 'string', 'enum', 'warnings'
    }

    # Known external packages
    external_packages = set()
    stdlib_found = set()

    for imp in imports:
        if imp in stdlib_modules:
            stdlib_found.add(imp)
        else:
            external_packages.add(imp)

    return external_packages, stdlib_found

def get_package_mapping():
    """Map import names to pip package names."""
    mapping = {
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'bidi': 'python-bidi',
        'arabic_reshaper': 'arabic-reshaper'
    }
    return mapping

def main():
    root_dir = Path(__file__).parent
    print(f"Analyzing dependencies in: {root_dir}")

    all_imports, file_imports = find_all_dependencies(root_dir)
    external_packages, stdlib_found = classify_imports(all_imports)

    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Total imports found: {len(all_imports)}")
    print(f"Standard library: {len(stdlib_found)}")
    print(f"External packages: {len(external_packages)}")

    print(f"\n=== EXTERNAL PACKAGES FOUND ===")
    package_mapping = get_package_mapping()

    pip_packages = set()
    for pkg in sorted(external_packages):
        pip_name = package_mapping.get(pkg, pkg)
        pip_packages.add(pip_name)
        print(f"  {pkg} -> {pip_name}")

    print(f"\n=== RECOMMENDED REQUIREMENTS.TXT ADDITIONS ===")
    # Read current requirements
    try:
        with open('requirements.txt', 'r') as f:
            current_reqs = f.read().lower()
    except FileNotFoundError:
        current_reqs = ""

    missing_packages = []
    for pkg in sorted(pip_packages):
        if pkg.lower() not in current_reqs:
            missing_packages.append(pkg)

    if missing_packages:
        print("Missing packages that should be added:")
        for pkg in missing_packages:
            print(f"  {pkg}")
    else:
        print("All packages appear to be in requirements.txt")

    print(f"\n=== DETAILED FILE BREAKDOWN ===")
    for file_path, imports in file_imports.items():
        rel_path = os.path.relpath(file_path, root_dir)
        external_in_file = {imp for imp in imports if imp not in stdlib_found}
        if external_in_file:
            print(f"\n{rel_path}:")
            for imp in sorted(external_in_file):
                print(f"  - {imp}")

if __name__ == "__main__":
    main()