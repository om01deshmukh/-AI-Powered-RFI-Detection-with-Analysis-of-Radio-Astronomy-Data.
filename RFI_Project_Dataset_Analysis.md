# RFI Detection Project - Dataset Analysis & Implementation Guide

## 📊 Dataset Overview

### Dataset 1: Kaggle RF Signal Image Classification
**Source:** https://www.kaggle.com/datasets/halcy0nic/radio-frequecy-rf-signal-image-classification

**Type:** Image-based RF signal classification dataset

**Potential Use Cases:**
- Pre-trained feature extraction for signal patterns
- Transfer learning for RFI detection
- Visual representation techniques
- Classification model baseline

**Expected Format:** 
- Spectrograms or waterfall plots as images
- Multiple signal classes (likely modulation types)
- Labeled training data

---

### Dataset 2: GMRT Observation Metadata (Your CSV)
**File:** `cart_items__1_.csv`

**Contents:**
```
- Description: Project codes and observation numbers
- LTA File Size (MB): Size of raw FITS files
- File Type: Raw FITS format
- Days Left: Archive availability
- LTA Status: Download/archive status
- Calibrated Status: Calibration completion
- Image Products Status: Processing status
- FITS Status: File conversion status
```

**Current Entry:**
- Project: 01SAK00, Observation: 361
- Size: 1901.29 MB (~1.9 GB)
- Status: Downloaded, FITS conversion successful


---

## 🎯 How These Datasets Fit Your Project

### Phase 1: Understanding Signal Patterns (Kaggle Dataset)
✅ **Use the Kaggle dataset first** to:
1. Learn RF signal visualization techniques
2. Build initial classification models
3. Understand feature extraction from spectrograms
4. Practice with manageable file sizes
5. Develop your UI/visualization framework

### Phase 2: Real RFI Detection (GMRT Data)
✅ **Transition to GMRT data** for:
1. Authentic radio astronomy signals
2. Real-world RFI patterns
3. Publication-quality results
4. GMRT-specific demonstrations

---

## 🛠️ Implementation Roadmap

### Step 1: Setup & Exploration (Week 1-2)

**A. Kaggle Dataset Exploration**
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load and visualize RF signal images
def explore_rf_images(image_dir):
    """
    Explore RF signal images from Kaggle dataset
    """
    for img_file in os.listdir(image_dir)[:5]:
        img = Image.open(os.path.join(image_dir, img_file))
        plt.figure(figsize=(10, 4))
        plt.imshow(img)
        plt.title(f'RF Signal: {img_file}')
        plt.colorbar()
        plt.show()

# Extract features for classification
def extract_features(image_path):
    """
    Extract statistical features from RF images
    """
    img = np.array(Image.open(image_path))
    
    features = {
        'mean_intensity': np.mean(img),
        'std_intensity': np.std(img),
        'max_intensity': np.max(img),
        'energy': np.sum(img**2),
        'peak_to_average': np.max(img) / (np.mean(img) + 1e-10)
    }
    return features
```

**B. GMRT FITS File Handling**
```python
from astropy.io import fits
import numpy as np

def load_gmrt_fits(fits_file):
    """
    Load GMRT FITS file and extract visibility data
    """
    with fits.open(fits_file) as hdul:
        # Print file structure
        hdul.info()
        
        # Access visibility data (typically in PRIMARY HDU)
        data = hdul[0].data
        header = hdul[0].header
        
        return data, header

def create_dynamic_spectrum(data):
    """
    Create time vs frequency plot (dynamic spectrum)
    """
    # Assuming data shape: (time, frequency, polarization)
    # Average over polarizations
    spectrum = np.mean(np.abs(data), axis=2) if data.ndim == 3 else np.abs(data)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrum, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Frequency Channel')
    plt.ylabel('Time')
    plt.title('Dynamic Spectrum')
    plt.colorbar(label='Intensity')
    plt.show()
    
    return spectrum
```

---

### Step 2: RFI Detection Model (Week 3-4)

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class RFIDetector:
    """
    AI-based RFI detection using anomaly detection
    """
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def extract_features(self, spectrum):
        """
        Extract features for RFI detection
        """
        features = []
        
        for time_slice in spectrum:
            # Statistical features
            feat = [
                np.mean(time_slice),
                np.std(time_slice),
                np.max(time_slice),
                np.median(time_slice),
                np.percentile(time_slice, 95),
                # Kurtosis (peaked-ness)
                np.mean((time_slice - np.mean(time_slice))**4) / (np.std(time_slice)**4 + 1e-10),
                # Skewness
                np.mean((time_slice - np.mean(time_slice))**3) / (np.std(time_slice)**3 + 1e-10)
            ]
            features.append(feat)
            
        return np.array(features)
    
    def fit(self, clean_spectrum):
        """
        Train on clean data
        """
        features = self.extract_features(clean_spectrum)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        
    def predict(self, spectrum):
        """
        Detect RFI in spectrum
        Returns: mask (1 = clean, -1 = RFI)
        """
        features = self.extract_features(spectrum)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return predictions
    
    def clean_data(self, spectrum, mask):
        """
        Remove RFI-flagged data
        """
        clean_spectrum = spectrum.copy()
        clean_spectrum[mask == -1] = np.nan  # or interpolate
        return clean_spectrum
```

---

### Step 3: Explainable AI Module (Week 5)

```python
import matplotlib.pyplot as plt
import seaborn as sns

class RFIExplainer:
    """
    Explain why signals are flagged as RFI
    """
    def __init__(self, detector):
        self.detector = detector
        
    def explain_detection(self, spectrum, time_idx):
        """
        Explain why a specific time slice was flagged
        """
        time_slice = spectrum[time_idx]
        features = self.detector.extract_features(spectrum[time_idx:time_idx+1])[0]
        
        feature_names = [
            'Mean Intensity',
            'Std Deviation',
            'Max Intensity',
            'Median',
            '95th Percentile',
            'Kurtosis',
            'Skewness'
        ]
        
        # Create explanation plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Signal plot with RFI markers
        axes[0, 0].plot(time_slice)
        axes[0, 0].set_title('Signal at Time Slice')
        axes[0, 0].set_xlabel('Frequency Channel')
        axes[0, 0].set_ylabel('Intensity')
        
        # Mark high peaks (potential RFI)
        threshold = np.percentile(time_slice, 95)
        rfi_channels = np.where(time_slice > threshold)[0]
        axes[0, 0].scatter(rfi_channels, time_slice[rfi_channels], 
                          color='red', s=50, label='Potential RFI', zorder=5)
        axes[0, 0].legend()
        
        # 2. Feature importance
        axes[0, 1].barh(feature_names, features, color='skyblue')
        axes[0, 1].set_title('Extracted Features')
        axes[0, 1].set_xlabel('Value')
        
        # 3. Histogram comparison
        axes[1, 0].hist(time_slice, bins=50, alpha=0.7, color='blue')
        axes[1, 0].axvline(np.mean(time_slice), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(time_slice):.2f}')
        axes[1, 0].axvline(threshold, color='orange', 
                          linestyle='--', label=f'95th %ile: {threshold:.2f}')
        axes[1, 0].set_title('Intensity Distribution')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Explanation text
        explanation = self._generate_explanation(features, feature_names)
        axes[1, 1].text(0.1, 0.5, explanation, fontsize=12, 
                       verticalalignment='center', wrap=True)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Detection Explanation')
        
        plt.tight_layout()
        plt.show()
        
    def _generate_explanation(self, features, feature_names):
        """
        Generate human-readable explanation
        """
        mean, std, max_val, median, p95, kurtosis, skewness = features
        
        reasons = []
        
        if max_val > 3 * mean:
            reasons.append(f"⚠️ Extremely high peak detected ({max_val:.2f} vs avg {mean:.2f})")
        
        if kurtosis > 5:
            reasons.append(f"⚠️ Sharp spikes present (kurtosis: {kurtosis:.2f})")
            
        if skewness > 2:
            reasons.append(f"⚠️ Asymmetric distribution (skewness: {skewness:.2f})")
            
        if std > mean:
            reasons.append(f"⚠️ High variability (std: {std:.2f})")
        
        if not reasons:
            return "✅ Signal appears clean\n- No unusual patterns detected\n- Normal statistical properties"
        else:
            return "🚫 RFI Detected:\n\n" + "\n\n".join(reasons)
    
    def compare_before_after(self, original, cleaned):
        """
        Visual comparison of raw vs cleaned data
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original
        im1 = axes[0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Original Data (With RFI)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Frequency Channel')
        axes[0].set_ylabel('Time')
        plt.colorbar(im1, ax=axes[0])
        
        # Cleaned
        im2 = axes[1].imshow(cleaned, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Cleaned Data', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Frequency Channel')
        axes[1].set_ylabel('Time')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference (RFI mask)
        difference = np.abs(original - cleaned)
        im3 = axes[2].imshow(difference, aspect='auto', origin='lower', cmap='Reds')
        axes[2].set_title('Removed RFI', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Frequency Channel')
        axes[2].set_ylabel('Time')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
```

---

### Step 4: Interactive Dashboard (Week 6)

```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("🔭 Radio Astronomy RFI Detection Tool")
    st.markdown("### AI-Powered Explainable RFI Detection")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload FITS or Image file", 
        type=['fits', 'fit', 'png', 'jpg']
    )
    
    # Detection sensitivity
    sensitivity = st.sidebar.slider(
        "RFI Detection Sensitivity",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01
    )
    
    if uploaded_file is not None:
        # Load data (simplified)
        if uploaded_file.name.endswith('.fits'):
            # Load FITS
            st.info("Loading FITS file...")
            # data, header = load_gmrt_fits(uploaded_file)
            # For demo, create synthetic data
            data = generate_synthetic_data()
        else:
            # Load image
            from PIL import Image
            img = Image.open(uploaded_file)
            data = np.array(img)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Raw Data", 
            "🔍 RFI Detection", 
            "✨ Cleaned Data", 
            "📚 Explanation"
        ])
        
        # Initialize detector
        detector = RFIDetector(contamination=sensitivity)
        
        with tab1:
            st.subheader("Original Radio Signal")
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('Frequency Channel')
            ax.set_ylabel('Time')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Intensity", f"{np.mean(data):.2f}")
            col2.metric("Max Intensity", f"{np.max(data):.2f}")
            col3.metric("Std Deviation", f"{np.std(data):.2f}")
        
        with tab2:
            st.subheader("RFI Detection Results")
            
            # Run detection
            predictions = detector.predict(data)
            rfi_percentage = (np.sum(predictions == -1) / len(predictions)) * 100
            
            st.warning(f"🚨 RFI detected in {rfi_percentage:.1f}% of time samples")
            
            # Visualize RFI mask
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Color code: green = clean, red = RFI
            colors = np.where(predictions.reshape(-1, 1) == 1, 0, 1)
            colors = np.repeat(colors, data.shape[1], axis=1)
            
            im = ax.imshow(colors, aspect='auto', origin='lower', 
                          cmap='RdYlGn_r', alpha=0.5)
            ax.set_xlabel('Frequency Channel')
            ax.set_ylabel('Time')
            ax.set_title('RFI Mask (Red = RFI, Green = Clean)')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Cleaned Radio Signal")
            
            # Clean data
            cleaned_data = detector.clean_data(data, predictions)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(cleaned_data, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('Frequency Channel')
            ax.set_ylabel('Time')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            
            # Improvement metrics
            col1, col2 = st.columns(2)
            noise_reduction = ((np.std(data) - np.nanstd(cleaned_data)) / np.std(data)) * 100
            col1.metric("Noise Reduction", f"{noise_reduction:.1f}%")
            col2.metric("Data Quality Score", f"{100 - rfi_percentage:.1f}%")
        
        with tab4:
            st.subheader("🧠 Explainable AI - Why was RFI detected?")
            
            # Select time slice to explain
            time_idx = st.slider(
                "Select time slice to examine",
                0, len(data)-1, 
                int(np.where(predictions == -1)[0][0]) if np.any(predictions == -1) else 0
            )
            
            # Generate explanation
            explainer = RFIExplainer(detector)
            
            st.markdown(f"### Analysis of Time Slice {time_idx}")
            
            # Show explanation visualization
            time_slice = data[time_idx]
            features = detector.extract_features(data[time_idx:time_idx+1])[0]
            
            # Create explanation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Signal Characteristics")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(time_slice)
                threshold = np.percentile(time_slice, 95)
                ax.axhline(threshold, color='red', linestyle='--', label='RFI Threshold')
                ax.set_xlabel('Frequency Channel')
                ax.set_ylabel('Intensity')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Detection Reasoning")
                
                mean, std, max_val, median, p95, kurtosis, skewness = features
                
                if predictions[time_idx] == -1:
                    st.error("🚫 **RFI DETECTED**")
                    st.markdown("**Reasons:**")
                    
                    if max_val > 3 * mean:
                        st.markdown(f"- ⚠️ Extreme spike: {max_val:.2f} (3x mean: {3*mean:.2f})")
                    if kurtosis > 5:
                        st.markdown(f"- ⚠️ Sharp peaks detected (kurtosis: {kurtosis:.2f})")
                    if skewness > 2:
                        st.markdown(f"- ⚠️ Asymmetric pattern (skewness: {skewness:.2f})")
                else:
                    st.success("✅ **CLEAN SIGNAL**")
                    st.markdown("No suspicious patterns detected")
                
                # Show feature values
                st.markdown("---")
                st.markdown("**Feature Values:**")
                st.write(f"Mean: {mean:.2f}")
                st.write(f"Std Dev: {std:.2f}")
                st.write(f"Max: {max_val:.2f}")
                st.write(f"Kurtosis: {kurtosis:.2f}")

def generate_synthetic_data():
    """
    Generate synthetic radio data with RFI for demo
    """
    np.random.seed(42)
    
    # Create base noise
    data = np.random.randn(100, 256) * 10 + 50
    
    # Add some RFI spikes
    for _ in range(10):
        t = np.random.randint(0, 100)
        f_start = np.random.randint(0, 200)
        f_width = np.random.randint(10, 50)
        data[t, f_start:f_start+f_width] += np.random.rand() * 200
    
    # Add broadband RFI
    rfi_times = np.random.choice(100, 15, replace=False)
    data[rfi_times, :] += 100
    
    return np.maximum(data, 0)  # Ensure non-negative

if __name__ == "__main__":
    main()
```

To run: `streamlit run app.py`

---

## 📋 Next Steps Checklist

### Immediate Actions (This Week)
- [ ] Download Kaggle RF dataset
- [ ] Set up Python environment with required libraries
- [ ] Create basic visualization script
- [ ] Test FITS file loading with GMRT data

### Short Term (2-3 Weeks)
- [ ] Implement RFI detection model
- [ ] Build explainability module
- [ ] Create interactive dashboard
- [ ] Test with both datasets

### Medium Term (1-2 Months)
- [ ] Optimize detection algorithms
- [ ] Add pulsar detection module
- [ ] Create documentation and tutorials
- [ ] Prepare demonstration for GMRT/Science Day

---

## 🔧 Required Python Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install astropy  # For FITS files
pip install streamlit  # For dashboard
pip install plotly  # Interactive plots
pip install pillow  # Image processing
```

For deep learning (optional):
```bash
pip install torch torchvision  # PyTorch
# or
pip install tensorflow  # TensorFlow
```

---

## 📚 Additional Resources

### RFI Detection Papers
- "Machine Learning for Radio Astronomy" (various papers)
- GMRT RFI mitigation documentation
- SKA RFI flagging strategies

### Tutorials
- Astropy FITS tutorial
- Scikit-learn anomaly detection
- Streamlit dashboard building

### Datasets
- HTRU Pulsar Dataset
- LOFAR RFI data
- MeerKAT open data

---

## 🎓 Educational Value

This project is perfect for:
- **Science exhibitions** - Interactive demo with clear visualations
- **Student workshops** - Hands-on ML + astronomy
- **Research demonstrations** - Real GMRT applications
- **Public outreach** - Making radio astronomy accessible

---

## ✨ Key Differentiators

1. **Visual & Interactive** - Not just numbers, but clear plots
2. **Explainable** - Shows *why* RFI was detected
3. **Educational** - Built for learning, not just research
4. **Extensible** - Easy to add pulsar detection module
5. **GMRT-Relevant** - Uses actual observatory data

---

*Ready to start coding? Begin with the Kaggle dataset exploration and gradually move to GMRT data!* 🚀
