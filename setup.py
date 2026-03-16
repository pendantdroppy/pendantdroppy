from setuptools import setup, find_packages
import os
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Create .desktop file for Linux
def create_desktop_file():
    """Create a .desktop file for Linux applications menu."""
    desktop_content = """[Desktop Entry]
Version=1.0
Type=Application
Name=droppy
Comment=Tensiometry software for droplet shape analysis
Exec=droppy
Icon=droppy
Terminal=false
Categories=Science;Physics;
Keywords=droplet;tensiometry;analysis;young-laplace;pendant;
"""
    
    # Determine install location
    if sys.platform == "linux" or sys.platform == "linux2":
        home = os.path.expanduser("~")
        desktop_dir = os.path.join(home, ".local/share/applications")
        desktop_file = os.path.join(desktop_dir, "droppy.desktop")
        
        try:
            os.makedirs(desktop_dir, exist_ok=True)
            with open(desktop_file, "w") as f:
                f.write(desktop_content)
            # Make it executable readable
            os.chmod(desktop_file, 0o644)
            print(f"✓ Created desktop file at {desktop_file}")
            print(f"  Launch with: droppy (command will be in PATH)")
            print(f"  Or click 'droppy' in your applications menu")
        except Exception as e:
            print(f"! Could not create desktop file: {e}")

setup(
    name="pendantdroppy",
    version="1.0.0",
    author="Josh",
    author_email="c0cz89m5t@mozmail.com",
    description="pendantdroppy - Tensiometry software for droplet shape analysis using Young-Laplace fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pendantdroppy/pendantdroppy",
    project_urls={
        "Bug Tracker": "https://github.com/pendantdroppy/pendantdroppy/issues",
        "Documentation": "https://github.com/pendantdroppy/pendantdroppy#readme",
        "Source Code": "https://github.com/pendantdroppy/pendantdroppy",
    },
    py_modules=["droppy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "PyQt6>=6.0.0",
    ],
    entry_points={
        "gui_scripts": [
            "droppy=droppy:main",
        ],
    },
    include_package_data=True,
    package_data={
        ".": ["*.conf", "*.png", "*.svg"],  # Include config and icon files in root
    },
    zip_safe=False,
    keywords="droplet tensiometry analysis young-laplace bond-number surface-tension pendant",
)

# Create desktop file after installation
create_desktop_file()



