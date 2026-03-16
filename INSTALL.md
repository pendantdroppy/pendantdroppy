# Installation Guide for pendantdroppy

## Quick Start (Recommended)

### Windows

1. **Install Python** (if not already installed)
   - Download from https://www.python.org/downloads/
   - During installation, CHECK "Add Python to PATH"

2. **Open Command Prompt** and run:
   ```bash
   pip install pendantdroppy
   droppy
   ```

### macOS

1. **Install Python** (if not already installed)
   ```bash
   # Using Homebrew (recommended)
   brew install python3
   ```

2. **Open Terminal** and run:
   ```bash
   pip3 install pendantdroppy
   droppy
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip python3-tk
   ```

2. **Install pendantdroppy**
   ```bash
   pip3 install pendantdroppy
   droppy
   ```

   The installation will automatically create a `droppy.desktop` file in your applications menu (or just run `droppy` from terminal).

---

## From Source (Development)

### 1. Clone Repository
```bash
git clone https://github.com/pendantdroppy/pendantdroppy.git
cd droppy
```

### 2. Create Virtual Environment (Optional but Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install in Development Mode
```bash
pip install -e .
```

### 5. Run the Application
```bash
droppy
```

Or directly:
```bash
python droppy.py
```

---

## Troubleshooting

### "pip: command not found"
- Ensure Python is installed and added to PATH
- Try `pip3` instead of `pip` (especially on macOS/Linux)
- Verify: `python --version`

### "ModuleNotFoundError: No module named 'PyQt6'"
```bash
pip install PyQt6
```

### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### "Permission denied" on macOS/Linux
```bash
pip install --user droppy
```

### OpenCV issues on macOS with Apple Silicon
```bash
# If using conda
conda install opencv

# Or ensure you have the right architecture
pip install --upgrade --force-reinstall opencv-python
```

### GUI doesn't open or crashes
- Update PyQt6: `pip install --upgrade PyQt6`
- Check display settings (especially on remote systems)
- Ensure OpenGL drivers are up to date

### Command not found: droppy
- Ensure installation completed: `pip install pendantdroppy`
- Try: `python -m droppy.gui`

### .desktop file not created on Linux
- This is created automatically during installation
- Check `~/.local/share/applications/droppy.desktop`
- If missing, you can create it manually from the terminal

---

## Manual Installation (No pip)

If you have issues with pip, install manually:

1. **Download required packages:**
   - Download `droppy.py`
   - Install dependencies: OpenCV, NumPy, PyQt6

2. **Run directly:**
   ```bash
   python droppy.py
   ```

---

## Docker Installation (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "droppy.py"]
```

Build and run:
```bash
docker build -t droppy .
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix droppy
```

---

## Verify Installation

```bash
# Check if installed correctly
python -c "import droppy; print('Success!')"

# Or try to import the GUI
python -c "from droppy.gui import MainWindow; print('GUI module loaded')"
```

---

## Upgrading

```bash
pip install --upgrade droppy
```

---

## Uninstalling

```bash
pip uninstall droppy
```

---

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum
- **Disk:** 500MB (including dependencies)
- **Display:** 1024x768 minimum resolution

## Dependencies

- **numpy** - Numerical computations
- **opencv-python** - Image processing
- **PyQt6** - GUI framework

---

## Getting Help

- Check README.md for usage instructions
- Review parameter explanations in README
- Visit: https://github.com/pendantdroppy/pendantdroppy/issues

---

## Next Steps

After installation, see **README.md** for:
- Usage workflow
- Parameter explanations
- Output file descriptions
- Troubleshooting common issues
