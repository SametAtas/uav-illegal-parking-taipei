import re

# Make sure this matches your image name
img_path = 'DJI_0518.JPG' 

print("--- Drone Flight Data ---")
with open(img_path, 'rb') as f:
    # Read the raw binary data and decode it to text
    img_data = f.read().decode('latin-1')

# Search for the DJI XMP tags
tags_to_find =[
    'GpsLatitude', 'GpsLongitude', 'RelativeAltitude', 
    'GimbalPitchDegree', 'GimbalRollDegree', 'GimbalYawDegree',
    'FlightPitchDegree', 'FlightRollDegree', 'FlightYawDegree'
]

# Find and print the data
for tag in tags_to_find:
    match = re.search(fr'drone-dji:{tag}="([^"]+)"', img_data)
    if match:
        print(f"{tag}: {match.group(1)}")
    else:
        print(f"{tag}: NOT FOUND")