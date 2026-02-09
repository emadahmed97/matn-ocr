#!/usr/bin/env python3
"""
Basic environment test that checks core structure without heavy ML dependencies.
Use this for local testing before pushing to HuggingFace Spaces.
"""

import sys
import os
import importlib.util
from pathlib import Path

def test_basic_python_imports():
    """Test basic Python and lightweight imports."""
    print("🔍 Testing Basic Dependencies...")

    basic_imports = [
        ("os", "Operating System Interface"),
        ("sys", "System-specific parameters"),
        ("json", "JSON encoder/decoder"),
        ("pathlib", "Object-oriented filesystem paths"),
        ("typing", "Type hints"),
        ("logging", "Logging facility"),
        ("tempfile", "Temporary file operations"),
        ("datetime", "Date and time handling"),
        ("importlib", "Import machinery"),
    ]

    failed = 0
    for module, desc in basic_imports:
        try:
            __import__(module)
            print(f"✅ {module} - {desc}")
        except ImportError as e:
            print(f"❌ {module} - {desc}: {e}")
            failed += 1

    return failed

def test_available_packages():
    """Test which ML packages are already available."""
    print("\n🔍 Testing Available ML Packages...")

    ml_packages = [
        "torch", "numpy", "PIL", "requests", "mlflow",
        "arabic_reshaper", "fastapi", "transformers",
        "gradio", "datasets", "accelerate", "peft"
    ]

    available = []
    unavailable = []

    for package in ml_packages:
        try:
            __import__(package)
            available.append(package)
            print(f"✅ {package}")
        except ImportError:
            unavailable.append(package)
            print(f"⚠️ {package} - Not available")

    print(f"\n📊 Summary: {len(available)}/{len(ml_packages)} packages available")

    if unavailable:
        print(f"\n📝 Missing packages for HF Spaces:")
        for pkg in unavailable:
            print(f"  - {pkg}")

    return len(unavailable)

def test_file_structure():
    """Test that all required files exist."""
    print("\n🔍 Testing File Structure...")

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "mlflow_arabic_ocr_config.py",
        "pipelines/__init__.py",
        "pipelines/arabic_ocr/__init__.py",
        "pipelines/arabic_ocr_training_pipeline.py",
    ]

    missing = []
    current_dir = Path(__file__).parent

    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing.append(file_path)

    if missing:
        print(f"\n📝 Missing files:")
        for file in missing:
            print(f"  - {file}")

    return len(missing)

def test_syntax_validation():
    """Test that Python files have valid syntax."""
    print("\n🔍 Testing Python Syntax...")

    current_dir = Path(__file__).parent
    python_files = []

    # Find all Python files
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(os.path.join(root, file))

    syntax_errors = 0
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            rel_path = os.path.relpath(file_path, current_dir)
            print(f"✅ {rel_path}")
        except SyntaxError as e:
            rel_path = os.path.relpath(file_path, current_dir)
            print(f"❌ {rel_path} - Syntax error: {e}")
            syntax_errors += 1
        except Exception as e:
            rel_path = os.path.relpath(file_path, current_dir)
            print(f"⚠️ {rel_path} - Error reading: {e}")

    return syntax_errors

def test_requirements_format():
    """Test that requirements.txt is properly formatted."""
    print("\n🔍 Testing Requirements.txt Format...")

    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()

        print(f"✅ requirements.txt found ({len(lines)} lines)")

        # Count non-comment, non-empty lines
        actual_deps = [line.strip() for line in lines
                      if line.strip() and not line.strip().startswith('#')]

        print(f"✅ {len(actual_deps)} dependencies listed")

        # Check for known problematic packages
        problematic = []
        for line in actual_deps:
            if any(pkg in line.lower() for pkg in ['xformers', 'bitsandbytes']):
                if not line.strip().startswith('#'):
                    problematic.append(line.strip())

        if problematic:
            print(f"⚠️ Found build-heavy dependencies (may fail locally):")
            for dep in problematic:
                print(f"  - {dep}")

        return 0

    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return 1

def test_readme_format():
    """Test that README.md has proper HuggingFace Spaces format."""
    print("\n🔍 Testing README.md Format...")

    try:
        with open('README.md', 'r') as f:
            content = f.read()

        print("✅ README.md found")

        # Check for YAML frontmatter
        if content.startswith('---'):
            print("✅ YAML frontmatter detected")

            # Check for required HF Spaces fields
            required_fields = ['title:', 'sdk:', 'app_file:', 'hardware:']
            missing_fields = []

            for field in required_fields:
                if field in content:
                    print(f"✅ {field} found")
                else:
                    missing_fields.append(field)
                    print(f"❌ {field} missing")

            return len(missing_fields)
        else:
            print("❌ No YAML frontmatter - required for HuggingFace Spaces")
            return 1

    except FileNotFoundError:
        print("❌ README.md not found")
        return 1

def main():
    """Run all basic tests."""
    print("🧪 Basic Environment Test for Arabic OCR Training")
    print("=" * 60)

    total_issues = 0

    total_issues += test_basic_python_imports()
    total_issues += test_available_packages()
    total_issues += test_file_structure()
    total_issues += test_syntax_validation()
    total_issues += test_requirements_format()
    total_issues += test_readme_format()

    print("\n" + "=" * 60)
    if total_issues == 0:
        print("🎉 All basic tests passed! Environment structure looks good.")
        print("📝 Note: Some ML packages may install differently on HuggingFace Spaces.")
        print("🚀 Ready to push to HuggingFace Spaces!")
        return 0
    else:
        print(f"⚠️ {total_issues} issue(s) found. Review before deploying.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)