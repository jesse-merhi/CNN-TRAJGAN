import argparse
import csv
import statistics

# Function to find the maximum, minimum, mean, and median latitudes and longitudes
def find_max_min_mean_median_lat_lon(input_file):
    latitudes = []
    longitudes = []

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for line in reader:
            lat, lon = float(line[2]), float(line[3])
            latitudes.append(lat)
            longitudes.append(lon)

    max_lat = max(latitudes)
    min_lat = min(latitudes)
    max_lon = max(longitudes)
    min_lon = min(longitudes)

    mean_lat = statistics.mean(latitudes)
    mean_lon = statistics.mean(longitudes)

    return max_lat, min_lat, max_lon, min_lon, mean_lat, mean_lon

# Input file path

parser = argparse.ArgumentParser()
parser.add_argument("file", help="First file to be combined.")


# Parse args
args = parser.parse_args()

max_lat, min_lat, max_lon, min_lon, mean_lat, mean_lon = find_max_min_mean_median_lat_lon(args.file )

print("Maximum Latitude:", max_lat)
print("Minimum Latitude:", min_lat)
print("Maximum Longitude:", max_lon)
print("Minimum Longitude:", min_lon)
print("Mean Latitude:", mean_lat)
print("Mean Longitude:", mean_lon)
