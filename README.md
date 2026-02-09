# 🔭 AI-Powered RFI Detection Tool for Radio Astronomy

An explainable AI tool for visualizing and cleaning Radio Frequency Interference (RFI) in radio telescope data, designed for GMRT and educational purposes.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

This tool makes RFI detection in radio astronomy data:
- **Visual** - Clear before/after comparisons
- **Explainable** - Shows *why* signals are flagged as interference
- **Educational** - Designed for students and science exhibitions
- **AI-Powered** - Uses machine learning for intelligent detection

### Key Features
✅ Automatic RFI detection using Isolation Forest  
✅ Visual explanations of detection reasoning  
✅ Interactive plots and comparisons  
✅ Support for GMRT FITS files  
✅ Compatible with Kaggle RF datasets  
✅ Ready for Streamlit dashboard deployment  

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this repository
# cd rfi-detection-tool

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
# Run with synthetic data (no dataset needed!)
python rfi_detection_starter.py
```

This will:
- Generate synthetic radio data with RFI
- Detect interference using AI
- Create detailed visualizations
- Explain why signals were flagged
- Save results as PNG images

### 3. Expected Output

The script will create:
- `rfi_detection_results.png` - Comprehensive comparison plot
- Interactive plots showing:
  - Original data with RFI
  - RFI detection mask
  - Cleaned data
  - Statistical analysis
  - Feature explanations

---

## 📊 Using Your Own Data

### Option 1: GMRT FITS Files

```python
from rfi_detection_starter import RFIDetector, load_gmrt_fits

# Load your FITS file
spectrum, header = load_gmrt_fits('your_observation.fits')

# Detect RFI
detector = RFIDetector(contamination=0.15)
predictions = detector.fit_predict(spectrum)

# Clean data
cleaned = detector.clean_spectrum(spectrum, predictions)
```

### Option 2: Kaggle RF Dataset

1. Download from: https://www.kaggle.com/datasets/halcy0nic/radio-frequecy-rf-signal-image-classification

2. Load images:
```python
from PIL import Image
import numpy as np

# Load RF signal image
img = Image.open('path/to/rf_signal.png')
spectrum = np.array(img)

# If image is 2D grayscale, use directly
# If RGB, convert: spectrum = np.mean(spectrum, axis=2)

# Run detection
detector = RFIDetector()
predictions = detector.fit_predict(spectrum)
```

---

## 🎓 Understanding the Results

### What is RFI?
Radio Frequency Interference (RFI) is unwanted electromagnetic radiation that contaminates astronomical signals. Sources include:
- Mobile towers 📱
- Satellites 🛰️
- Power lines ⚡
- Electronic devices 💻
- Aircraft radar ✈️

### How Detection Works
The tool extracts 10 statistical features from each time slice:
1. **Mean intensity** - Average signal level
2. **Standard deviation** - Signal variability
3. **Maximum value** - Peak intensity
4. **Median** - Middle value
5. **95th percentile** - High intensity threshold
6. **5th percentile** - Low intensity threshold
7. **Range** - Max - Min
8. **Kurtosis** - "Peakedness" (>5 indicates sharp spikes)
9. **Skewness** - Asymmetry (>2 indicates bias)
10. **Peak-to-average ratio** - Spike detection (>5 is suspicious)

AI (Isolation Forest) learns normal patterns and flags anomalies as RFI.

### Interpretation Guide

**🚫 RFI Indicators:**
- Peak-to-average ratio > 5
- Kurtosis > 5 (sharp spikes)
- Skewness > 2 (asymmetric)
- Max > 3× mean (extreme peaks)

**✅ Clean Signal:**
- Normal statistical properties
- Low kurtosis (2-3)
- Low skewness (-1 to 1)
- Reasonable peak-to-average ratio (2-3)

---

## 🎨 Customization

### Adjust Detection Sensitivity

```python
# More aggressive (detect more RFI)
detector = RFIDetector(contamination=0.20)  # 20% expected RFI

# Conservative (only obvious RFI)
detector = RFIDetector(contamination=0.05)  # 5% expected RFI

# Default
detector = RFIDetector(contamination=0.10)  # 10% expected RFI
```

### Choose Cleaning Method

```python
# Set RFI to zero
cleaned = detector.clean_spectrum(spectrum, predictions, method='zero')

# Set RFI to NaN (for statistics)
cleaned = detector.clean_spectrum(spectrum, predictions, method='nan')

# Interpolate (smooth replacement)
cleaned = detector.clean_spectrum(spectrum, predictions, method='interpolate')
```

---

## 📈 Next Steps & Extensions

### Phase 1: Current ✅
- [x] RFI detection with Isolation Forest
- [x] Visual comparisons
- [x] Feature-based explanations
- [x] Support for synthetic data

### Phase 2: In Progress 🚧
- [ ] Interactive Streamlit dashboard
- [ ] Real-time parameter tuning
- [ ] Multiple visualization modes
- [ ] Export cleaned FITS files

### Phase 3: Future 🔮
- [ ] Pulsar detection module
- [ ] Deep learning models (CNN/RNN)
- [ ] Real-time RFI monitoring
- [ ] Integration with SDR hardware

---

## 🛠️ Troubleshooting

### Common Issues

**1. ImportError: No module named 'astropy'**
```bash
pip install astropy
```

**2. FITS file not loading**
- Check file path is correct
- Ensure file is valid FITS format
- Try opening with `astropy.io.fits.open()` manually

**3. Memory errors with large files**
```python
# Load subset of data
with fits.open('large_file.fits') as hdul:
    data = hdul[0].data[0:1000, :]  # First 1000 time samples
```

**4. Poor detection results**
- Adjust `contamination` parameter
- Check if data needs normalization
- Verify data shape (should be 2D: time × frequency)

---

## 📚 Educational Resources

### For Students
- **What is radio astronomy?** Study of cosmic radio waves
- **Why clean RFI?** To see weak astronomical signals
- **Machine learning basics** Classification, anomaly detection
- **Feature engineering** Extracting meaningful patterns

### For Researchers
- Paper: "Machine Learning for Radio Astronomy" (various)
- GMRT RFI documentation
- SKA RFI strategies
- Pulsar detection methods

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional feature extraction methods
- Deep learning models
- New visualization techniques
- Documentation improvements
- Real GMRT data examples

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍🔬 About

Developed for:
- **GMRT** (Giant Metrewave Radio Telescope)
- **Science Day demonstrations**
- **Educational workshops**
- **Student research projects**

**Goal:** Make radio astronomy data processing accessible and understandable!

---

## 📧 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Project: GMRT RFI Detection Tool

---

## 🙏 Acknowledgments

- GMRT team for inspiration
- Kaggle for RF datasets
- scikit-learn for ML tools
- Python astronomy community

---

**Happy RFI hunting! 🔭✨**
