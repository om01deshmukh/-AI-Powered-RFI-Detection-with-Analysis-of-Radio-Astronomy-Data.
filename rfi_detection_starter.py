"""
RFI Detection Tool - Starter Script
AI-Powered Visual & Explainable RFI Detection for Radio Astronomy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA GENERATION (Replace with your actual data loading)
# ============================================================================

def generate_synthetic_rfi_data(n_time=100, n_freq=256, rfi_count=15):
    """
    Generate synthetic radio data with RFI for testing
    
    In production, replace this with:
    - load_gmrt_fits() for GMRT data
    - load_kaggle_image() for Kaggle RF dataset
    """
    np.random.seed(42)
    
    # Create base cosmic signal (low noise floor)
    data = np.random.randn(n_time, n_freq) * 5 + 50
    
    # Add weak astronomical signal
    for i in range(5):
        t_center = np.random.randint(20, 80)
        f_center = np.random.randint(50, 200)
        for t in range(n_time):
            for f in range(n_freq):
                signal = 30 * np.exp(-((t-t_center)**2 + (f-f_center)**2) / 200)
                data[t, f] += signal
    
    # Add RFI - narrowband spikes
    for _ in range(rfi_count // 3):
        t = np.random.randint(0, n_time)
        f_start = np.random.randint(0, n_freq - 30)
        f_width = np.random.randint(5, 30)
        data[t, f_start:f_start+f_width] += np.random.uniform(100, 300)
    
    # Add RFI - broadband interference
    for _ in range(rfi_count // 3):
        t_start = np.random.randint(0, n_time - 5)
        t_width = np.random.randint(1, 5)
        data[t_start:t_start+t_width, :] += np.random.uniform(50, 150)
    
    # Add RFI - persistent frequency channel
    for _ in range(rfi_count // 3):
        f = np.random.randint(0, n_freq)
        data[:, f] += np.random.uniform(80, 200)
    
    return np.maximum(data, 0)  # Ensure non-negative


def load_gmrt_fits(fits_file):
    """
    Load GMRT FITS file
    Requires: pip install astropy
    """
    from astropy.io import fits
    
    with fits.open(fits_file) as hdul:
        print("FITS file structure:")
        hdul.info()
        
        # Get visibility data
        data = hdul[0].data
        header = hdul[0].header
        
        # Extract dynamic spectrum (time vs frequency)
        # This depends on your specific FITS structure
        if data.ndim >= 2:
            spectrum = np.mean(np.abs(data), axis=-1) if data.ndim > 2 else np.abs(data)
        else:
            raise ValueError("Unexpected FITS data structure")
        
        return spectrum, header


# ============================================================================
# 2. RFI DETECTOR CLASS
# ============================================================================

class RFIDetector:
    """
    AI-based RFI detection using Isolation Forest
    """
    
    def __init__(self, contamination=0.1, verbose=True):
        """
        Args:
            contamination: Expected proportion of RFI (0.0 to 0.5)
            verbose: Print detection statistics
        """
        self.contamination = contamination
        self.verbose = verbose
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, spectrum):
        """
        Extract statistical features from each time slice
        """
        n_time = spectrum.shape[0]
        features = []
        
        for t in range(n_time):
            time_slice = spectrum[t, :]
            
            # Statistical features
            feat = [
                np.mean(time_slice),                    # Mean intensity
                np.std(time_slice),                     # Standard deviation
                np.max(time_slice),                     # Maximum value
                np.median(time_slice),                  # Median
                np.percentile(time_slice, 95),          # 95th percentile
                np.percentile(time_slice, 5),           # 5th percentile
                np.max(time_slice) - np.min(time_slice), # Range
                # Kurtosis (peaked-ness, >3 indicates spikes)
                np.mean((time_slice - np.mean(time_slice))**4) / (np.std(time_slice)**4 + 1e-10),
                # Skewness (asymmetry)
                np.mean((time_slice - np.mean(time_slice))**3) / (np.std(time_slice)**3 + 1e-10),
                # Peak-to-average ratio
                np.max(time_slice) / (np.mean(time_slice) + 1e-10)
            ]
            
            features.append(feat)
        
        return np.array(features)
    
    def fit_predict(self, spectrum):
        """
        Fit model and predict RFI in one step
        """
        features = self.extract_features(spectrum)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit and predict
        predictions = self.model.fit_predict(features_scaled)
        self.is_fitted = True
        
        # Print statistics
        if self.verbose:
            n_rfi = np.sum(predictions == -1)
            n_total = len(predictions)
            print(f"\n{'='*60}")
            print(f"RFI Detection Results:")
            print(f"{'='*60}")
            print(f"Total time samples: {n_total}")
            print(f"RFI detected: {n_rfi} ({100*n_rfi/n_total:.1f}%)")
            print(f"Clean samples: {n_total - n_rfi} ({100*(n_total-n_rfi)/n_total:.1f}%)")
            print(f"{'='*60}\n")
        
        return predictions
    
    def clean_spectrum(self, spectrum, predictions, method='interpolate'):
        """
        Remove RFI from spectrum
        
        Args:
            spectrum: Input data
            predictions: RFI mask (1=clean, -1=RFI)
            method: 'zero', 'nan', or 'interpolate'
        """
        clean = spectrum.copy()
        rfi_mask = predictions == -1
        
        if method == 'zero':
            clean[rfi_mask, :] = 0
        elif method == 'nan':
            clean[rfi_mask, :] = np.nan
        elif method == 'interpolate':
            # Simple linear interpolation
            for f in range(clean.shape[1]):
                if np.any(rfi_mask):
                    # Find clean samples
                    clean_indices = np.where(~rfi_mask)[0]
                    rfi_indices = np.where(rfi_mask)[0]
                    
                    if len(clean_indices) >= 2:
                        # Interpolate
                        clean[rfi_indices, f] = np.interp(
                            rfi_indices,
                            clean_indices,
                            clean[clean_indices, f]
                        )
        
        return clean


# ============================================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_comparison(original, cleaned, predictions, save_fig=False):
    """
    Create before/after comparison plot
    """
    fig = plt.figure(figsize=(18, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Original spectrum
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(original, aspect='auto', origin='lower', 
                     cmap='viridis', interpolation='nearest')
    ax1.set_title('Original Data (With RFI)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frequency Channel')
    ax1.set_ylabel('Time Sample')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # 2. RFI mask
    ax2 = fig.add_subplot(gs[0, 2])
    rfi_visual = np.where(predictions.reshape(-1, 1) == 1, 0, 1)
    rfi_visual = np.repeat(rfi_visual, original.shape[1], axis=1)
    im2 = ax2.imshow(rfi_visual, aspect='auto', origin='lower', 
                     cmap='RdYlGn_r', interpolation='nearest')
    ax2.set_title('RFI Mask', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Time')
    
    # 3. Cleaned spectrum
    ax3 = fig.add_subplot(gs[1, :2])
    im3 = ax3.imshow(cleaned, aspect='auto', origin='lower', 
                     cmap='viridis', interpolation='nearest')
    ax3.set_title('Cleaned Data (RFI Removed)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frequency Channel')
    ax3.set_ylabel('Time Sample')
    plt.colorbar(im3, ax=ax3, label='Intensity')
    
    # 4. Statistics
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # Calculate metrics
    n_rfi = np.sum(predictions == -1)
    n_total = len(predictions)
    noise_orig = np.std(original)
    noise_clean = np.nanstd(cleaned)
    noise_reduction = ((noise_orig - noise_clean) / noise_orig) * 100
    
    stats_text = f"""
    STATISTICS
    {'='*25}
    
    RFI Detection:
    • Total samples: {n_total}
    • RFI detected: {n_rfi}
    • RFI percentage: {100*n_rfi/n_total:.1f}%
    
    Data Quality:
    • Original σ: {noise_orig:.2f}
    • Cleaned σ: {noise_clean:.2f}
    • Noise reduction: {noise_reduction:.1f}%
    
    Signal Range:
    • Original: [{np.min(original):.1f}, {np.max(original):.1f}]
    • Cleaned: [{np.nanmin(cleaned):.1f}, {np.nanmax(cleaned):.1f}]
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    # 5. Difference (what was removed)
    ax5 = fig.add_subplot(gs[2, :2])
    difference = original - cleaned
    difference[predictions == 1] = 0  # Only show RFI regions
    im5 = ax5.imshow(difference, aspect='auto', origin='lower', 
                     cmap='Reds', interpolation='nearest')
    ax5.set_title('Removed RFI (Difference)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Frequency Channel')
    ax5.set_ylabel('Time Sample')
    plt.colorbar(im5, ax=ax5, label='Removed Intensity')
    
    # 6. Histogram comparison
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.hist(original.flatten(), bins=50, alpha=0.6, label='Original', color='red')
    ax6.hist(cleaned[~np.isnan(cleaned)].flatten(), bins=50, alpha=0.6, 
             label='Cleaned', color='green')
    ax6.set_xlabel('Intensity')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Intensity Distribution')
    ax6.legend()
    ax6.set_yscale('log')
    
    plt.suptitle('RFI Detection & Cleaning Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_fig:
        plt.savefig('rfi_detection_results.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'rfi_detection_results.png'")
    
    plt.show()


def plot_time_slice_explanation(spectrum, predictions, detector, time_idx):
    """
    Explain why a specific time slice was flagged as RFI
    """
    time_slice = spectrum[time_idx, :]
    features = detector.extract_features(spectrum[time_idx:time_idx+1])[0]
    is_rfi = predictions[time_idx] == -1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time slice plot
    axes[0, 0].plot(time_slice, linewidth=1.5)
    axes[0, 0].set_title(f'Time Slice {time_idx} - {"RFI DETECTED" if is_rfi else "CLEAN"}',
                         fontsize=12, fontweight='bold',
                         color='red' if is_rfi else 'green')
    axes[0, 0].set_xlabel('Frequency Channel')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark high peaks
    threshold = np.percentile(time_slice, 95)
    peak_channels = np.where(time_slice > threshold)[0]
    if len(peak_channels) > 0:
        axes[0, 0].scatter(peak_channels, time_slice[peak_channels],
                          color='red', s=30, zorder=5, alpha=0.6,
                          label=f'Peaks > 95th percentile')
        axes[0, 0].legend()
    
    # 2. Feature values
    feature_names = [
        'Mean', 'Std Dev', 'Max', 'Median',
        '95th %ile', '5th %ile', 'Range',
        'Kurtosis', 'Skewness', 'Peak/Avg'
    ]
    
    colors = ['red' if is_rfi else 'green'] * len(feature_names)
    axes[0, 1].barh(feature_names, features, color=colors, alpha=0.7)
    axes[0, 1].set_title('Extracted Features', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Histogram
    axes[1, 0].hist(time_slice, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(time_slice), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(time_slice):.2f}')
    axes[1, 0].axvline(np.median(time_slice), color='orange', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(time_slice):.2f}')
    axes[1, 0].set_title('Intensity Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Explanation text
    axes[1, 1].axis('off')
    
    mean, std, max_val, median, p95, p5, range_val, kurtosis, skewness, peak_avg = features
    
    if is_rfi:
        title = "🚫 RFI DETECTION EXPLANATION"
        color = 'red'
        
        reasons = []
        if peak_avg > 5:
            reasons.append(f"• Very high peak-to-average ratio: {peak_avg:.2f}")
        if max_val > 3 * mean:
            reasons.append(f"• Extreme spike: {max_val:.1f} vs mean {mean:.1f}")
        if kurtosis > 5:
            reasons.append(f"• Sharp peaks detected (kurtosis: {kurtosis:.2f})")
        if skewness > 2:
            reasons.append(f"• Highly asymmetric (skewness: {skewness:.2f})")
        if std > mean:
            reasons.append(f"• High variability (σ > μ)")
        
        explanation = f"{title}\n\n" + "\n".join(reasons) if reasons else \
                     f"{title}\n\nGeneral anomaly detected"
    else:
        title = "✅ CLEAN SIGNAL"
        color = 'green'
        explanation = f"""{title}

No significant RFI indicators found:
• Normal peak-to-average ratio: {peak_avg:.2f}
• Reasonable kurtosis: {kurtosis:.2f}
• Low skewness: {skewness:.2f}
• Expected statistical properties
"""
    
    axes[1, 1].text(0.1, 0.5, explanation, fontsize=11, family='monospace',
                   verticalalignment='center', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("="*60)
    print("RFI Detection Tool - AI-Powered Radio Astronomy")
    print("="*60)
    
    # Step 1: Generate/Load Data
    print("\n[1] Loading data...")
    spectrum = generate_synthetic_rfi_data(n_time=150, n_freq=256, rfi_count=25)
    print(f"Data shape: {spectrum.shape} (time x frequency)")
    print(f"Data range: [{np.min(spectrum):.2f}, {np.max(spectrum):.2f}]")
    
    # To use real GMRT data instead:
    # spectrum, header = load_gmrt_fits('your_observation.fits')
    
    # Step 2: Initialize Detector
    print("\n[2] Initializing RFI detector...")
    detector = RFIDetector(contamination=0.15, verbose=True)
    
    # Step 3: Detect RFI
    print("\n[3] Running RFI detection...")
    predictions = detector.fit_predict(spectrum)
    
    # Step 4: Clean Data
    print("\n[4] Cleaning data...")
    cleaned_spectrum = detector.clean_spectrum(spectrum, predictions, method='interpolate')
    
    # Step 5: Visualize Results
    print("\n[5] Creating visualizations...")
    print("\nGenerating comparison plot...")
    plot_comparison(spectrum, cleaned_spectrum, predictions, save_fig=True)
    
    # Step 6: Explain specific detections
    print("\n[6] Explaining RFI detections...")
    
    # Find first RFI sample
    rfi_indices = np.where(predictions == -1)[0]
    if len(rfi_indices) > 0:
        print(f"\nExplaining time slice {rfi_indices[0]} (RFI detected)...")
        plot_time_slice_explanation(spectrum, predictions, detector, rfi_indices[0])
    
    # Find first clean sample
    clean_indices = np.where(predictions == 1)[0]
    if len(clean_indices) > 0:
        print(f"\nExplaining time slice {clean_indices[0]} (Clean signal)...")
        plot_time_slice_explanation(spectrum, predictions, detector, clean_indices[0])
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Replace synthetic data with your GMRT FITS files")
    print("2. Tune contamination parameter (0.05-0.20)")
    print("3. Add more sophisticated features")
    print("4. Implement Streamlit dashboard for interactivity")
    print("5. Add pulsar detection module")
    print("="*60)


if __name__ == "__main__":
    main()
