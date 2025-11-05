#!/usr/bin/env python3
"""
ECOSTRESS Water Mask to Land Polygons

This script converts one or more ECOSTRESS Water Mask GeoTIFFs into a unified
set of land polygons clipped to an area of interest (AOI). It automatically
handles multiple input tiles from different UTM zones and reprojects everything
to EPSG:4326 (WGS84 latitude/longitude).

Inputs:
  - One or more ECOSTRESS water mask GeoTIFFs (e.g., ECO_L2T_WATERMASK_*.tif)
  - Area_of_Interest.shp in EPSG:4326

Outputs (saved in the input folder, all in EPSG:4326):
  - binary_summed.tif           (merged binary water mask)
  - binary_summed_clipped.tif   (clipped to AOI)
  - rigid.shp                   (unsmoothed land polygons)
  - smooth.shp, smoother.shp, smoothest.shp (progressively smoothed polygons)
"""

# Import the Libraries Needed for Polygonization
import os
import glob
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, Resampling
from rasterio.mask import mask
from rasterio.features import shapes
import fiona
from shapely.geometry import shape, mapping

# CONFIGURATION
INPUT_FOLDER = "<<< REPLACE_THIS_TEXT_WITH_FOLDER_PATH >>>"
AOI_SHP = os.path.join(INPUT_FOLDER, "<<< REPLACE_THIS_TEXT_WITH_AOI_NAME.shp >>>")

# MAIN FUNCTION
def main():
    print("\n=== ECOSTRESS Water Mask → Land Polygon Workflow ===\n")

    # Step 1: Gather all GeoTIFFs
    tiff_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    if not tiff_files:
        print("ERROR: No GeoTIFF files found in input folder.")
        return

    dst_crs = "EPSG:4326"
    valid_files = []
    all_bounds = []

    print(f"Found {len(tiff_files)} potential TIFFs. Checking validity...")

    # Step 2: Compute overall bounding box across all valid rasters
    for tif in tiff_files:
        try:
            with rasterio.open(tif) as src:
                b = transform_bounds(src.crs, dst_crs, *src.bounds)
                all_bounds.append(b)
                valid_files.append(tif)
        except Exception as e:
            print(f"Skipping {os.path.basename(tif)}: {e}")

    if not all_bounds:
        print("ERROR: No valid GeoTIFFs found.")
        return

    # Compute bounding box union
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    # Step 3: Define output grid parameters from first valid file
    with rasterio.open(valid_files[0]) as ref:
        ref_transform, ref_width, ref_height = calculate_default_transform(
            ref.crs, dst_crs, ref.width, ref.height, *ref.bounds
        )
        px_x = abs(ref_transform.a)
        px_y = abs(ref_transform.e)

    width = int((maxx - minx) / px_x)
    height = int((maxy - miny) / px_y)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": dst_crs,
        "transform": transform,
    }

    # Step 4: Reproject and combine all TIFFs
    print(f"\nMerging {len(valid_files)} ECOSTRESS tiles into one global grid...")
    data_sum = np.zeros((height, width), dtype=np.uint32)

    for tif in valid_files:
        with rasterio.open(tif) as src:
            arr = src.read(1)
            reprojected = np.zeros((height, width), dtype=np.uint32)
            reproject(
                source=arr,
                destination=reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            data_sum += (reprojected != 0).astype(np.uint32)
            print(f"   → Integrated {os.path.basename(tif)} ({src.crs.to_string()})")

    # Step 5: Threshold and save binary_summed.tif
    binary = (data_sum >= 1).astype(np.uint8)
    binary_path = os.path.join(INPUT_FOLDER, "binary_summed.tif")
    with rasterio.open(binary_path, "w", **out_meta) as dst:
        dst.write(binary, 1)
    print(f"\nSaved merged binary mask: {os.path.basename(binary_path)}")

    # Step 6: Clip to AOI
    if not os.path.exists(AOI_SHP):
        print(f"AOI shapefile not found: {AOI_SHP}")
        print("Skipping clip step.")
        return

    with fiona.open(AOI_SHP, "r") as shp:
        shapes_geom = [feat["geometry"] for feat in shp]

    with rasterio.open(binary_path) as src:
        clipped_img, clipped_transform = mask(src, shapes_geom, crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_img.shape[1],
            "width": clipped_img.shape[2],
            "transform": clipped_transform
        })

    clipped_path = os.path.join(INPUT_FOLDER, "binary_summed_clipped.tif")
    with rasterio.open(clipped_path, "w", **clipped_meta) as dst:
        dst.write(clipped_img)
    print(f"Saved clipped mask: {os.path.basename(clipped_path)}")

    # Step 7: Polygonize land (0 values = land)
    print("\nPolygonizing land areas...")
    with rasterio.open(clipped_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    land_mask = (arr == 0)
    raw_polys = [
        shape(geom)
        for geom, val in shapes(arr, mask=land_mask, transform=transform)
    ]

    schema = {"geometry": "Polygon", "properties": {}}
    rigid_path = os.path.join(INPUT_FOLDER, "rigid.shp")
    with fiona.open(rigid_path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        for poly in raw_polys:
            if not poly.is_valid:
                poly = poly.buffer(0)
            dst.write({"geometry": mapping(poly), "properties": {}})
    print("Saved rigid land polygons: rigid.shp")

    # Step 8: Generate smoothed polygon variants
    base_px = max(1e-8, min(abs(transform.a) or 0, abs(transform.e) or 0))
    smoothing_specs = [
        ("smooth.shp", 0.5),
        ("smoother.shp", 0.75),
        ("smoothest.shp", 1.0)
    ]

    for filename, factor in smoothing_specs:
        dist = base_px * factor
        smooth_path = os.path.join(INPUT_FOLDER, filename)
        with fiona.open(smooth_path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
            for poly in raw_polys:
                smoothed = poly.buffer(-dist).buffer(dist)
                final = smoothed if not smoothed.is_empty else poly
                if not final.is_valid:
                    final = final.buffer(0)
                dst.write({"geometry": mapping(final), "properties": {}})
        print(f"Saved {filename}")

    # Step 9: Done!
    print("\n Processing complete! Outputs saved in:")
    print(f"   {INPUT_FOLDER}\n")
    print("   • binary_summed.tif")
    print("   • binary_summed_clipped.tif")
    print("   • rigid.shp")
    print("   • smooth.shp")
    print("   • smoother.shp")
    print("   • smoothest.shp\n")

# RUN SCRIPT
if __name__ == "__main__":
    main()
