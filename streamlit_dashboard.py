#!/usr/bin/env python3
"""
Interactive RFI Detection Dashboard
Streamlit Web Application

Run with: streamlit run streamlit_dashboard.py
"""
# streamlit run streamlit_dashboard.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Import our RFI detector (make sure rfi_detection_starter.py is in same directory)
from rfi_detection_starter import (
    RFIDetector, 
    generate_synthetic_rfi_data,
    plot_comparison
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RFI Detection Tool",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_kaggle_image(uploaded_file):
    """Load and process uploaded image"""
    img = Image.open(uploaded_file)
    # Convert to grayscale if RGB
    if len(np.array(img).shape) == 3:
        img_array = np.mean(np.array(img), axis=2)
    else:
        img_array = np.array(img)
    return img_array


def create_matplotlib_figure_to_image(fig):
    """Convert matplotlib figure to image for streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🔭 Radio Astronomy RFI Detection Tool</p>', 
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Explainable RFI Detection for Radio Astronomy")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Data source selection
        st.subheader("1. Data Source")
        data_source = st.radio(
            "Select data type:",
            ["Demo (Synthetic)", "Upload FITS", "Upload Image"],
            help="Choose where to get the radio data from"
        )
        
        # File upload
        uploaded_file = None
        if data_source == "Upload FITS":
            uploaded_file = st.file_uploader(
                "Upload FITS file", 
                type=['fits', 'fit'],
                help="Upload your GMRT observation FITS file"
            )
        elif data_source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload RF Signal Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload spectrogram from Kaggle dataset"
            )
        
        st.markdown("---")
        
        # Detection parameters
        st.subheader("2. Detection Settings")
        
        contamination = st.slider(
            "RFI Sensitivity",
            min_value=0.01,
            max_value=0.50,
            value=0.15,
            step=0.01,
            help="Expected proportion of RFI (higher = more aggressive detection)"
        )
        
        cleaning_method = st.selectbox(
            "Cleaning Method",
            ["interpolate", "zero", "nan"],
            help="How to handle detected RFI"
        )
        
        # Demo data parameters
        if data_source == "Demo (Synthetic)":
            st.markdown("---")
            st.subheader("3. Demo Data Settings")
            
            n_time = st.slider("Time samples", 50, 200, 100)
            n_freq = st.slider("Frequency channels", 128, 512, 256)
            rfi_count = st.slider("RFI events", 5, 40, 20)
        
        st.markdown("---")
        
        # Run button
        run_detection = st.button("🚀 Run RFI Detection", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Info
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            This tool uses **Isolation Forest** (ML algorithm) to detect 
            Radio Frequency Interference in radio telescope data.
            
            **Features:**
            - Visual before/after comparison
            - AI-powered detection
            - Explainable results
            - Interactive exploration
            
            **Developed for:** GMRT & Science Exhibitions
            """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data
    if run_detection or st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            # Determine data source
            if data_source == "Demo (Synthetic)":
                spectrum = generate_synthetic_rfi_data(n_time, n_freq, rfi_count)
                st.success(f"✅ Generated synthetic data: {spectrum.shape}")
            
            elif data_source == "Upload FITS" and uploaded_file is not None:
                try:
                    # Load FITS file
                    from astropy.io import fits
                    with fits.open(uploaded_file) as hdul:
                        spectrum = np.abs(hdul[0].data)
                        if spectrum.ndim > 2:
                            spectrum = np.mean(spectrum, axis=-1)
                    st.success(f"✅ Loaded FITS file: {spectrum.shape}")
                except Exception as e:
                    st.error(f"Error loading FITS: {e}")
                    return
            
            elif data_source == "Upload Image" and uploaded_file is not None:
                spectrum = load_kaggle_image(uploaded_file)
                st.success(f"✅ Loaded image: {spectrum.shape}")
            
            else:
                st.warning("⚠️ Please upload a file or use demo data")
                return
            
            st.session_state.data_loaded = True
            st.session_state.spectrum = spectrum
        
        spectrum = st.session_state.spectrum
        
        # ====================================================================
        # RUN DETECTION
        # ====================================================================
        
        with st.spinner("Running RFI detection..."):
            # Initialize detector
            detector = RFIDetector(contamination=contamination, verbose=False)
            
            # Detect RFI
            predictions = detector.fit_predict(spectrum)
            
            # Clean data
            cleaned_spectrum = detector.clean_spectrum(
                spectrum, predictions, method=cleaning_method
            )
            
            # Calculate metrics
            n_rfi = np.sum(predictions == -1)
            n_total = len(predictions)
            rfi_percentage = (n_rfi / n_total) * 100
            
            noise_orig = np.std(spectrum)
            noise_clean = np.nanstd(cleaned_spectrum)
            noise_reduction = ((noise_orig - noise_clean) / noise_orig) * 100
        
        st.success("✅ Detection complete!")
        
        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🔍 Original Data",
            "✨ Cleaned Data",
            "📈 Comparison",
            "🧠 Explanation"
        ])
        
        # --- TAB 1: Overview ---
        with tab1:
            st.header("Detection Summary")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Samples",
                    n_total,
                    help="Number of time samples analyzed"
                )
            
            with col2:
                st.metric(
                    "RFI Detected",
                    f"{n_rfi} ({rfi_percentage:.1f}%)",
                    delta=f"-{n_rfi}",
                    delta_color="inverse",
                    help="Number and percentage of RFI samples"
                )
            
            with col3:
                st.metric(
                    "Noise Reduction",
                    f"{noise_reduction:.1f}%",
                    delta=f"{noise_reduction:.1f}%",
                    help="Reduction in standard deviation"
                )
            
            with col4:
                quality_score = 100 - rfi_percentage
                st.metric(
                    "Data Quality",
                    f"{quality_score:.1f}%",
                    delta=f"{rfi_percentage:.1f}%",
                    delta_color="inverse",
                    help="Percentage of clean data"
                )
            
            st.markdown("---")
            
            # Quick stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Data Statistics")
                stats_df_orig = {
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Range"],
                    "Value": [
                        f"{np.mean(spectrum):.2f}",
                        f"{np.median(spectrum):.2f}",
                        f"{np.std(spectrum):.2f}",
                        f"{np.min(spectrum):.2f}",
                        f"{np.max(spectrum):.2f}",
                        f"{np.max(spectrum) - np.min(spectrum):.2f}"
                    ]
                }
                st.dataframe(stats_df_orig, hide_index=True)
            
            with col2:
                st.subheader("Cleaned Data Statistics")
                stats_df_clean = {
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Range"],
                    "Value": [
                        f"{np.nanmean(cleaned_spectrum):.2f}",
                        f"{np.nanmedian(cleaned_spectrum):.2f}",
                        f"{np.nanstd(cleaned_spectrum):.2f}",
                        f"{np.nanmin(cleaned_spectrum):.2f}",
                        f"{np.nanmax(cleaned_spectrum):.2f}",
                        f"{np.nanmax(cleaned_spectrum) - np.nanmin(cleaned_spectrum):.2f}"
                    ]
                }
                st.dataframe(stats_df_clean, hide_index=True)
        
        # --- TAB 2: Original Data ---
        with tab2:
            st.header("Original Radio Signal (With RFI)")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(spectrum, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='nearest')
            ax.set_xlabel('Frequency Channel', fontsize=12)
            ax.set_ylabel('Time Sample', fontsize=12)
            ax.set_title('Dynamic Spectrum', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Intensity')
            st.pyplot(fig)
            plt.close()
            
            # Show RFI mask overlay
            st.subheader("RFI Detection Overlay")
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot spectrum
            im = ax.imshow(spectrum, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='nearest', alpha=0.7)
            
            # Overlay RFI mask
            rfi_visual = np.where(predictions.reshape(-1, 1) == -1, 1, np.nan)
            rfi_visual = np.repeat(rfi_visual, spectrum.shape[1], axis=1)
            ax.imshow(rfi_visual, aspect='auto', origin='lower',
                     cmap='Reds', alpha=0.5, interpolation='nearest')
            
            ax.set_xlabel('Frequency Channel', fontsize=12)
            ax.set_ylabel('Time Sample', fontsize=12)
            ax.set_title('Original Data with RFI Highlighted (Red)', 
                        fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Intensity')
            st.pyplot(fig)
            plt.close()
        
        # --- TAB 3: Cleaned Data ---
        with tab3:
            st.header("Cleaned Radio Signal (RFI Removed)")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(cleaned_spectrum, aspect='auto', origin='lower',
                          cmap='viridis', interpolation='nearest')
            ax.set_xlabel('Frequency Channel', fontsize=12)
            ax.set_ylabel('Time Sample', fontsize=12)
            ax.set_title('Cleaned Dynamic Spectrum', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Intensity')
            st.pyplot(fig)
            plt.close()
            
            st.success(f"✅ Successfully removed {n_rfi} RFI-contaminated time samples!")
        
        # --- TAB 4: Comparison ---
        with tab4:
            st.header("Before vs After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Before (With RFI)")
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(spectrum, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Time')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("After (Cleaned)")
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(cleaned_spectrum, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Time')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                plt.close()
            
            # Difference plot
            st.subheader("Removed RFI")
            difference = spectrum - cleaned_spectrum
            difference[predictions == 1] = 0
            
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(difference, aspect='auto', origin='lower', cmap='Reds')
            ax.set_xlabel('Frequency Channel', fontsize=12)
            ax.set_ylabel('Time Sample', fontsize=12)
            ax.set_title('What Was Removed (Red = RFI)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Removed Intensity')
            st.pyplot(fig)
            plt.close()
        
        # --- TAB 5: Explanation ---
        with tab5:
            st.header("🧠 Explainable AI - Understanding the Detection")
            
            st.markdown("""
            This section explains **why** the AI flagged certain signals as RFI.
            Select a time slice below to see detailed analysis.
            """)
            
            # Time slice selector
            time_idx = st.slider(
                "Select time slice to examine:",
                0, len(spectrum) - 1,
                int(np.where(predictions == -1)[0][0]) if np.any(predictions == -1) else 0,
                help="Choose a time sample to analyze in detail"
            )
            
            # Extract features and explain
            time_slice = spectrum[time_idx, :]
            features = detector.extract_features(spectrum[time_idx:time_idx+1])[0]
            is_rfi = predictions[time_idx] == -1
            
            # Display status
            if is_rfi:
                st.error(f"🚫 **Time Slice {time_idx}: RFI DETECTED**")
            else:
                st.success(f"✅ **Time Slice {time_idx}: CLEAN SIGNAL**")
            
            st.markdown("---")
            
            # Two columns for plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Signal Profile")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(time_slice, linewidth=1.5, color='blue')
                
                # Mark peaks
                threshold = np.percentile(time_slice, 95)
                peak_channels = np.where(time_slice > threshold)[0]
                if len(peak_channels) > 0:
                    ax.scatter(peak_channels, time_slice[peak_channels],
                              color='red', s=30, zorder=5, alpha=0.6,
                              label='Peaks > 95th %ile')
                    ax.legend()
                
                ax.set_xlabel('Frequency Channel')
                ax.set_ylabel('Intensity')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Feature values
                st.subheader("Extracted Features")
                feature_names = [
                    'Mean', 'Std Dev', 'Max', 'Median',
                    '95th %ile', '5th %ile', 'Range',
                    'Kurtosis', 'Skewness', 'Peak/Avg'
                ]
                
                feature_df = {
                    "Feature": feature_names,
                    "Value": [f"{f:.2f}" for f in features]
                }
                st.dataframe(feature_df, hide_index=True)
            
            with col2:
                st.subheader("Intensity Distribution")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(time_slice, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(np.mean(time_slice), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(time_slice):.2f}')
                ax.axvline(np.median(time_slice), color='orange', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(time_slice):.2f}')
                ax.set_xlabel('Intensity')
                ax.set_ylabel('Count')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Explanation
                st.subheader("Detection Reasoning")
                
                mean, std, max_val, median, p95, p5, range_val, kurtosis, skewness, peak_avg = features
                
                if is_rfi:
                    st.markdown("**🚫 Why this was flagged as RFI:**")
                    
                    reasons = []
                    if peak_avg > 5:
                        reasons.append(f"⚠️ Very high peak-to-average ratio: **{peak_avg:.2f}**")
                    if max_val > 3 * mean:
                        reasons.append(f"⚠️ Extreme spike: **{max_val:.1f}** vs mean **{mean:.1f}**")
                    if kurtosis > 5:
                        reasons.append(f"⚠️ Sharp peaks (kurtosis: **{kurtosis:.2f}**)")
                    if skewness > 2:
                        reasons.append(f"⚠️ Highly asymmetric (skewness: **{skewness:.2f}**)")
                    if std > mean:
                        reasons.append(f"⚠️ High variability (σ > μ)")
                    
                    for reason in reasons:
                        st.markdown(reason)
                    
                    if not reasons:
                        st.markdown("General anomaly detected by AI model")
                else:
                    st.markdown("**✅ Why this is clean:**")
                    st.markdown(f"- Normal peak-to-average: **{peak_avg:.2f}**")
                    st.markdown(f"- Reasonable kurtosis: **{kurtosis:.2f}**")
                    st.markdown(f"- Low skewness: **{skewness:.2f}**")
                    st.markdown("- Expected statistical properties")
        
        # ====================================================================
        # DOWNLOAD SECTION
        # ====================================================================
        
        st.markdown("---")
        st.header("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download cleaned data as CSV
            cleaned_csv = np.savetxt(
                "cleaned_spectrum.csv",
                cleaned_spectrum,
                delimiter=","
            )
            st.download_button(
                "Download Cleaned Data (CSV)",
                data=open("cleaned_spectrum.csv", "rb"),
                file_name="cleaned_spectrum.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download RFI mask
            mask_csv = np.savetxt(
                "rfi_mask.csv",
                predictions.reshape(-1, 1),
                delimiter=",",
                fmt='%d'
            )
            st.download_button(
                "Download RFI Mask (CSV)",
                data=open("rfi_mask.csv", "rb"),
                file_name="rfi_mask.csv",
                mime="text/csv"
            )


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
