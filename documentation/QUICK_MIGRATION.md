# 🚀 Quick Migration: pip → uv

**TL;DR**: Fast migration from pip to uv for Python projects.

## ⚡ Quick Steps

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Remove old environment
rm -rf venv  # or .venv

# 3. Create new uv environment
uv venv
source .venv/bin/activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Create lock file
uv pip compile requirements.txt --output-file requirements.lock

# 6. Test
python -c "import your_module; print('✅ Success!')"
```

## 📝 Update .gitignore

Add these lines:
```gitignore
.venv/
requirements.lock
build/
dist/
*.egg-info/
```

## 🔧 Optional: Add pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "your-project"
version = "1.0.0"
dependencies = [
    # Copy from requirements.txt
]

[project.scripts]
your-command = "your_module:main"  # Optional CLI
```

Then install in editable mode:
```bash
uv pip install -e .
```

## 📖 Update README

Replace pip instructions with:
```markdown
## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .  # or: uv pip install -r requirements.txt
```

## 🎯 Daily Commands

```bash
# Activate
source .venv/bin/activate

# Install new package
uv pip install package_name

# Update lock file
uv pip compile requirements.txt --output-file requirements.lock

# Reproduce exact environment
uv pip sync requirements.lock
```

## ✅ Benefits

- 🚀 **10-100x faster** than pip
- 🔒 **Reproducible builds** with lock files
- 🛠️ **Better dependency resolution**
- 📦 **Modern packaging** with pyproject.toml
- 🎯 **CLI tools** with entry points

---

**Done!** Your project now uses modern uv tooling. 🎉 