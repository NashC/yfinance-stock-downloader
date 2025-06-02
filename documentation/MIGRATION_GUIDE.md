# 📦 Complete Guide: Migrate Python Project from `pip` to `uv`

This guide provides step-by-step instructions for migrating an existing Python project from using `pip` and `venv` to the modern `uv` toolchain.

## 🎯 Overview

**What this migration accomplishes:**
- Faster dependency resolution and installation
- Modern Python packaging standards
- Reproducible builds with lock files
- Better dependency conflict resolution
- Optional command-line entry points
- Development tools integration

## 📋 Prerequisites

Before starting, ensure you have:
- An existing Python project with `requirements.txt`
- Basic familiarity with command line
- Git repository (recommended for tracking changes)

## 🚀 Migration Steps

### Step 1: Install uv (if not already installed)

```bash
# Install uv globally
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Step 2: Backup and Remove Old Environment

```bash
# Backup old environment (optional)
mv venv venv_backup  # or mv .venv .venv_backup

# Or remove completely
rm -rf venv  # or rm -rf .venv
```

### Step 3: Create New uv Environment

```bash
# Create new virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### Step 4: Install Dependencies with uv

```bash
# Install from existing requirements.txt
uv pip install -r requirements.txt
```

### Step 5: Generate Lock File for Reproducibility

```bash
# Create lock file with exact versions
uv pip compile requirements.txt --output-file requirements.lock
```

### Step 6: Update .gitignore

Add or update your `.gitignore` file:

```gitignore
# Virtual environments
.venv/
venv/
ENV/
env/

# uv
requirements.lock

# Build system (if using pyproject.toml)
build/
dist/
*.egg-info/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### Step 7: Create pyproject.toml (Optional but Recommended)

Create a `pyproject.toml` file for modern Python packaging:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "your-project-name"
version = "1.0.0"
description = "Your project description"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = [
    "keyword1",
    "keyword2",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.9"
dependencies = [
    # Copy from requirements.txt
    "pandas>=2.0.0",
    "requests>=2.25.0",
    # Add other dependencies...
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
]

# Optional: Add command-line entry point
[project.scripts]
your-command = "your_main_module:main"

[tool.black]
line-length = 88
target-version = ['py39', 'py310', 'py311', 'py312']

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q"
testpaths = ["tests"]

[tool.ruff]
target-version = "py39"
line-length = 88
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501"]
```

### Step 8: Install Project in Editable Mode (if using pyproject.toml)

```bash
# Install project with all dependencies
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

### Step 9: Update Documentation

Update your README.md with new setup instructions:

```markdown
## Setup

### Modern Setup with uv (Recommended)

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install project**:
   ```bash
   uv pip install -e .
   ```

### Alternative Installation Methods

**Install dependencies only**:
```bash
uv pip install -r requirements.txt
```

**Use lock file for exact reproducibility**:
```bash
uv pip sync requirements.lock
```

**Install with development tools**:
```bash
uv pip install -e ".[dev]"
```

### Legacy pip setup (if needed)
```bash
pip install -r requirements.txt
```
```

### Step 10: Test the Migration

```bash
# Verify environment works
python -c "import your_main_module; print('✅ Import successful!')"

# If you added a command-line entry point, test it
your-command --help  # or whatever your command is

# Run your tests
pytest  # if you have tests
```

### Step 11: Update CI/CD (if applicable)

Update your CI/CD configuration to use uv:

**GitHub Actions example:**
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: |
    uv venv
    source .venv/bin/activate
    uv pip sync requirements.lock
```

## 🎯 Post-Migration Workflow

### Daily Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Install new package
uv pip install package_name

# Update requirements.txt (if needed)
echo "package_name>=1.0.0" >> requirements.txt

# Update lock file
uv pip compile requirements.txt --output-file requirements.lock

# Install in editable mode
uv pip install -e .

# Run development tools (if configured)
black .
isort .
mypy .
ruff check .
pytest
```

### Team Collaboration

```bash
# For new team members or fresh environments
uv venv
source .venv/bin/activate
uv pip sync requirements.lock  # Exact reproducibility

# For development
uv pip install -e ".[dev]"
```

## 🔧 Troubleshooting

### Common Issues

**Problem**: `uv` command not found
**Solution**: 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart terminal or source shell profile
```

**Problem**: Import errors after migration
**Solution**:
```bash
uv pip install -e .  # Reinstall in editable mode
uv pip list  # Verify all packages installed
```

**Problem**: Command-line entry point not working
**Solution**:
- Ensure `pyproject.toml` has correct `[project.scripts]` section
- Verify main function exists in specified module
- Reinstall: `uv pip install -e .`

**Problem**: Dependencies not resolving
**Solution**:
```bash
# Clear and reinstall
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 📊 Benefits Achieved

After migration, you'll have:

✅ **Faster installs** - uv is significantly faster than pip  
✅ **Better dependency resolution** - More reliable conflict resolution  
✅ **Reproducible builds** - Lock file ensures exact versions  
✅ **Modern tooling** - Future-proof Python package management  
✅ **Optional CLI tools** - Easy command-line interface creation  
✅ **Development workflow** - Integrated linting, testing, formatting  

## 🎯 Next Steps

Consider these additional improvements:

1. **Add pre-commit hooks** for code quality
2. **Set up automated testing** with pytest
3. **Configure code formatting** with black and isort
4. **Add type checking** with mypy
5. **Set up continuous integration** with GitHub Actions
6. **Consider publishing** to PyPI if it's a reusable package

## 📚 Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 518 - pyproject.toml](https://peps.python.org/pep-0518/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)

---

**Migration Complete!** 🎉 Your project now uses modern Python packaging with uv. 