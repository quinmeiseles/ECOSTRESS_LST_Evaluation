#!/usr/bin/env python3
"""
NDBC Buoy Water Temperature Retrieval & Hourly Averaging Script

This script retrieves hourly-averaged buoy water temperature data from the
National Data Buoy Center (NDBC) API for all stations within a given Area of Interest (AOI).

How to use:
- Update all placeholder text marked as <<< REPLACE_THIS_TEXT >>> for your specific study area.
- Follow the instructions in each step carefully to adapt this workflow to your own location and time range.
"""

# Import the Libraries Needed for Retrieval/Binning
import os
import geopandas as gpd
from shapely.geometry import Point
from ndbc_api import NdbcApi
import pandas as pd
from datetime import timedelta
import pytz

# 1. CONFIGURATION
# Folder where your shapefile is stored and where your CSV will be saved
INPUT_DIRECTORY = '<<< REPLACE_THIS_TEXT_WITH_FOLDER_PATH >>>'

# Shapefile defining your AOI (Area of Interest)
SHAPEFILE_PATH = os.path.join(INPUT_DIRECTORY, '<<< REPLACE_THIS_TEXT_WITH_SHAPEFILE_NAME.shp >>>')

# Output CSV name (averaged buoy temperature results)
OUTPUT_CSV = os.path.join(INPUT_DIRECTORY, '<<< REPLACE_THIS_TEXT_WITH_OUTPUT_FILENAME.csv >>>')

# Date range for retrieval (inclusive)
START_DATE = '<<< REPLACE_THIS_TEXT_WITH_START_DATE (YYYY-MM-DD) >>>'  # e.g., '2020-01-01'
END_DATE = '<<< REPLACE_THIS_TEXT_WITH_END_DATE (YYYY-MM-DD) >>>'      # e.g., '2023-12-31'

# Modes for data retrieval - "ocean" and "stdmet" are the two main NDBC data categories where buoys report water temperature and related measurements
MODES_TO_TRY = ['ocean', 'stdmet']

# Possible names for water temperature columns in NDBC datasets
WTMP_COLS_CANDIDATES = ['WTMP', 'WTMP1', 'WTMP2', 'WTMP_M']

# Convert UTC to the local timezone for your study area (example: US/Eastern)
utc = pytz.utc
local_tz = pytz.timezone("<<< REPLACE_THIS_TEXT_WITH_TIMEZONE >>>")

# 2. LOAD SHAPEFILE
print("[INFO] Loading shapefile...")
aoi = gpd.read_file(SHAPEFILE_PATH)
print(f"[INFO] AOI original CRS: {aoi.crs}")

# Convert to WGS84 if necessary
aoi = aoi.to_crs("EPSG:4326")

# 3. FETCH NDBC STATIONS AND FILTER BY AOI
print("[INFO] Fetching NDBC stations list from API...")
api = NdbcApi()
stations_df = api.stations()

print("[INFO] Filtering stations by AOI boundary...")
points = [Point(lon, lat) for lat, lon in zip(stations_df['Lat'], stations_df['Lon'])]
stations_geo = gpd.GeoDataFrame(stations_df, geometry=points, crs="EPSG:4326")
stations_in_aoi = gpd.sjoin(stations_geo, aoi, how="inner", predicate='intersects')

if stations_in_aoi.empty:
    print("[ERROR] No NDBC stations found in AOI.")
    print("[DEBUG] Check shapefile's CRS and coordinates.")
    print(aoi.bounds)
    exit()

print(f"[INFO] Found {len(stations_in_aoi)} stations inside AOI.")
print("[DEBUG] Station IDs and coordinates:")
print(stations_in_aoi[['Station', 'Lat', 'Lon']])

# 4. RETRIEVE WATER TEMPERATURE DATA FOR EACH STATION
print("[INFO] Retrieving buoy data...")
all_records = []

for _, row in stations_in_aoi.iterrows():
    station_id = row['Station']
    lat = row['Lat']
    lon = row['Lon']
    station_data_collected = False

    for mode in MODES_TO_TRY:
        try:
            # Request buoy data
            data = api.get_data(
                station_id=station_id,
                mode=mode,
                start_time=START_DATE,
                end_time=END_DATE,
                as_df=True,
            )

            # Skip if no valid data
            if isinstance(data, dict) or data.empty:
                print(f"[INFO] No valid data for {station_id} mode {mode}.")
                continue

            # Identify the correct temperature column
            wtmp_col = next((col for col in WTMP_COLS_CANDIDATES if col in data.columns), None)
            if not wtmp_col:
                print(f"[INFO] No water temperature column found for {station_id} mode {mode}.")
                continue

            # Extract timestamps from index and convert to local timezone
            if isinstance(data.index, pd.MultiIndex):
                timestamps = pd.to_datetime(data.index.get_level_values(0))
            else:
                timestamps = pd.to_datetime(data.index)

            if timestamps.tz is None:
                timestamps = timestamps.tz_localize("UTC")
            timestamps = timestamps.tz_convert(local_tz)

            # Create a working DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'water_temperature': data[wtmp_col].values
            })

            df = df.dropna(subset=['water_temperature'])
            if df.empty:
                continue

            # Round timestamps to nearest hour
            def round_to_nearest_hour(ts):
                if ts.minute < 30:
                    return ts.replace(minute=0, second=0, microsecond=0)
                else:
                    return (ts + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

            df['rounded_timestamp'] = df['timestamp'].apply(round_to_nearest_hour)

            # Group by rounded hour and calculate average (excluding 2*std outliers if applicable)
            grouped = df.groupby('rounded_timestamp')
            for ts, group in grouped:
                temps = group['water_temperature']
                if len(temps) <= 2 or temps.std(skipna=True) == 0 or pd.isna(temps.std()):
                    avg_temp = temps.mean()
                else:
                    mean = temps.mean()
                    std = temps.std()
                    filtered = temps[(temps >= mean - 2 * std) & (temps <= mean + 2 * std)]
                    if filtered.empty:
                        continue
                    avg_temp = filtered.mean()

                all_records.append({
                    'timestamp': ts.strftime('%Y%m%d%H%M'),
                    'water_temperature': avg_temp,
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon
                })

            print(f"[SUCCESS] Data collected for {station_id} mode {mode}")
            station_data_collected = True
            break

        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {station_id} mode {mode}: {e}")

    if not station_data_collected:
        print(f"[WARNING] No water temperature data collected for {station_id}")

# 5. SAVE OUTPUT CSV
if all_records:
    final_df = pd.DataFrame(all_records)
    print(f"[INFO] Writing output CSV to {OUTPUT_CSV}")
    final_df.to_csv(OUTPUT_CSV, index=False)
else:
    print("[WARNING] No data collected from any stations.")

print("[DONE]")
