# Project 2 Data Aquisition using Fast Fouirer Transfrom

import pandas as pd
import serial
import numpy as np
import time
import os
import sys

# --- 0. User Input for Labeling ---
# Gives a label for each recroding
print("What experiment am I running right now?")
print("1 = LED is Off, 2 = 10Hz Flicker, 3 = 20Hz Flicker")
choice = input("Enter 1, 2, or 3: ")

if choice == "1":
    current_label = "off"
elif choice == "2":
    current_label = "10Hz"
else:
    current_label = "20Hz"

print(f"Recording started for: {current_label}")

# --- 1. Settings and Serial Connection ---
port = 'COM3'  # Change to your specific Arduino port
baud = 115200
T_record = 10  # Recording for 10 seconds per sample to ensure enough data for FFT

arduino = serial.Serial(port, baud, timeout=1)
time.sleep(2)  # Critical: Give the Arduino time to reset its serial buffer after opening

print(f'Collecting {T_record} seconds of data...')

data_list = []
start_time = time.time()

# Loop until the 10 second timer is finished
while (time.time() - start_time) < T_record:
    if arduino.in_waiting > 0:
        try:
            # Read the "ms,adc" string, decode from bytes, and split into parts
            line = arduino.readline().decode('utf-8').strip()
            parts = line.split(',')
            if len(parts) == 2:
                timestamp = float(parts[0])
                adc_value = float(parts[1])
                data_list.append([timestamp, adc_value])
        except:
            # Skip any corrupted lines that occurred during serial transmission
            pass

arduino.close()
print(f'Got {len(data_list)} data points.')

# --- 2. Signal Processing Pipeline ---
data = np.array(data_list)
t = data[:, 0] / 1000  # Convert millisecond timestamps to seconds
x = data[:, 1]         # Extract the raw sensor ADC values

# Calculate actual sampling rate (fs) from time delta
# I used the median difference over the mean difference so that random jitters will not affect the output as much
dt = np.diff(t)
dt = dt[dt > 0] # Filter out any non-positive deltas to avoid division by zero
fs = 1 / np.median(dt)
print(f'Calculated fs: {fs:.2f} Hz')

# --- 3. Feature Extraction (Frequency Domain) ---
# Remove the DC component (mean) before the FFT calculation
# This prevents the 0Hz peak from overwhelming the smaller flicker signals
x_centered = x - np.mean(x)

# I am using np.fft.rfft so that it returns only positive values
N = len(x_centered)
pos_mags = np.abs(np.fft.rfft(x_centered))
pos_freqs = np.fft.rfftfreq(N, d=1/fs)

# Dominant frequency - which frequency has the biggest magnitude
# Skip DC component at index 0 since we already removed the mean
dom_idx = np.argmax(pos_mags[1:]) + 1
dominant_freq = float(pos_freqs[dom_idx])

# Band power ratio - how much energy is in a specific frequency range
# I use an adaptive band based on the label to verify the target frequency is present
if current_label == "off":
    band_start, band_end = 0, 5  # Look for low frequency noise/drift
elif current_label == "10Hz":
    band_start, band_end = 8, 12  # Look for the 10 Hz target signal
else:  # 20Hz
    band_start, band_end = 18, 22  # Look for the 20 Hz target signal

# Find which FFT bins fall within our target frequency range
band_mask = (pos_freqs >= band_start) & (pos_freqs <= band_end)

# Power is magnitude squared (based on Parseval's theorem)
# I calculate the ratio of target power vs total power to see signal strength
band_power = np.sum(pos_mags[band_mask]**2)
total_power = np.sum(pos_mags[1:]**2)  # Exclude DC/0Hz from total power too
band_power_ratio = float(band_power / total_power)

# Solve for peak magnitude which is the strength of the dominant frequency spike
peak_magnitude = float(pos_mags[dom_idx])

# Spectral Centroid - This represents the "center of mass" of the spectrum
# Calculated by the sum of all frequencies weighted by their magnitudes, divided by total magnitude
spectral_centroid = float(np.sum(pos_freqs * pos_mags) / np.sum(pos_mags))

# Spectral Spread - how concentrated or "spread out" the spectrum is around the centroid
# Effectively the standard deviation of the spectrum relative to the centroid
spectral_spread = float(
    np.sqrt(np.sum(((pos_freqs - spectral_centroid)**2) * pos_mags) / np.sum(pos_mags))
)

print('\n--- Extracted Research Features ---')
print(f'Dominant Frequency: {dominant_freq:.2f} Hz')
print(f'Band Power Ratio ({band_start}-{band_end} Hz): {band_power_ratio:.4f}')
print(f'Peak Magnitude: {peak_magnitude:.3f}')
print(f'Spectral Centroid: {spectral_centroid:.2f}')
print(f'Spectral Spread: {spectral_spread:.2f}')

# Sanity check - dominant frequency should never be negative
if dominant_freq < 0:
    print("ERROR: Negative frequency detected! Something is wrong with FFT.")
    sys.exit()

# --- 4. Store and Save to CSV ---
# I organize these into a dictionary to ensure they map correctly to the CSV columns
features = {
    'label': current_label,
    'dominant_freq': round(dominant_freq, 2),
    'band_power_ratio': round(band_power_ratio, 4),
    'peak magnitude': round(peak_magnitude, 2),
    'spectral centroid': round(spectral_centroid, 2),
    'spectral spread': round(spectral_spread, 2),
}

# Define the file path for saving the dataset
save_path = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(save_path, exist_ok=True)

full_file = os.path.join(save_path, 'features.csv')

# Append the new row to the features file for later Machine Learning training
# header=False ensures we don't write the column names multiple times
df = pd.DataFrame([features])
df.to_csv(full_file, mode='a', header=not os.path.exists(full_file), index=False)

print(f'\nData saved to: {full_file}')
