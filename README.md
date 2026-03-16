# pendantdroppy - Tensiometry Analysis Software

A PyQt6-based GUI application for analyzing droplet shape using Young-Laplace equation fitting and Bond number calculation.

## Features

- **Interactive Geometry Selection** - cv2-based ROI and needle line drawing
- **Dual Clarity Analysis** - Runs low and high clarity fits, averages Bond number
- **Young-Laplace Fitting** - Theoretical curve matching with lensing correction
- **Configuration Saving** - Save/load analysis settings to `droppy.conf`
- **Visual Results** - YL curve overlay with plateau region highlighting
- **Comprehensive Output** - JSON summary, CSV data, PNG visualizations
- **Desktop Integration** - Automatic .desktop file creation on Linux

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### From PyPI (once published)
```bash
pip install droppy
```

### From Source
```bash
# Clone the repository
git clone https://github.com/pendantdroppy/pendantdroppy.git
cd droppy

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Manual Installation
```bash
# Install dependencies
pip install opencv-python PyQt6 numpy

# Run the GUI
python droppy.py
```

## Usage

### GUI Application
```bash
droppy
```

Or run directly:
```bash
python droppy.py
```

### Workflow
1. **Select Image** - Browse for droplet image (PNG, JPG, etc.)
2. **Draw Geometry** - Interactive cv2 window to define:
   - ROI (Region of Interest)
   - Needle line (for diameter calibration)
3. **Configure Parameters** - Adjust processing and analysis settings across tabs
4. **Save Settings** - Button to save current configuration to `droppy.conf`
5. **Run Analysis** - Click "RUN ANALYSIS" to:
   - Run low clarity fit
   - Run high clarity fit
   - Average Bond numbers
6. **View Results** - See overlaid YL curve, plateau points, and extrema

### Settings File
Configuration is automatically saved to `droplet_out/droppy.conf`. 

On startup, the program loads from this file if it exists, otherwise uses defaults.

Example `droppy.conf`:
```json
{
  "droplet_type": "rising",
  "needle_diameter_mm": 0.72,
  "sigma": 2.0,
  "canny1": 40,
  "canny2": 120,
  "plateau_width": 100,
  "lensing_factor": 1.0,
  "low_clarity_ratio": 0.1,
  "high_clarity_ratio": 0.01
}
```

### Output Files
Analysis generates:
- `result_yl_overlay.png` - Young-Laplace curve fitted to droplet
- `edges.png` - Edge detection result
- `summary.json` - Numerical results (Bond numbers, calibration)
- `droplet_edge_results.csv` - Detailed point-by-point analysis

## Parameters Explained

### Image Processing
- **Blur Sigma** - Gaussian blur for noise reduction
- **Canny Low/High** - Edge detection thresholds

### Contour Analysis
- **Num Points** - Number of contour points to sample
- **Direction** - Traversal direction (+1 or -1)
- **Min r', z'** - Minimum thresholds to avoid numerical blowup

### Analysis
- **Plateau Width** - Number of points centered on equator
- **Lensing Factor** - Optical correction (1.0 = no correction)
- **Clarity Ratios** - Window fractions for low/high clarity runs
- **Bo Wiggle Room** - Expansion of Bond number search range

## Visualization Colors

- **Red** - Young-Laplace theoretical curve (averaged Bond number)
- **Yellow** - Plateau region (fitting region around equator)
- **Grey** - Extrema points (tip, center, left, right)

## System Requirements

- Windows, macOS, or Linux
- 4GB RAM minimum
- OpenCV, PyQt6, NumPy

## Troubleshooting

**"No contours found"** - Adjust Canny thresholds or sigma
**"Bo ~0 for rising droplets"** - Ensure plateau is centered on equator; check lensing factor
**Config not loading** - Check `droppy.conf` is valid JSON in output directory

## License

MIT License - see LICENSE file

## Citation

If you use this software in research, please cite:

```bibtex
@software{droppy2024,
  title={dropPy - Tensiometry Analysis Software},
  author={Josh},
  year={2024},
  url={https://github.com/pendantdroppy/pendantdroppy}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing documentation
- Review analysis parameters

## Acknowledgments

Based on Young-Laplace equation analysis for axisymmetric droplets.

References:
- Bashforth, F., & Adams, J. C. (1883). "An attempt to test the theories of capillary action"
- Rotenberg, Y., Boruvka, L., & Neumann, A. W. (1983). "Determination of surface tension and contact angle from the shapes of axisymmetric fluid interfaces"
