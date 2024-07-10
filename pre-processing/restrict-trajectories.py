import csv
import argparse
# Function to filter and save full trajectories within specified bounds
def filter_and_save_trajectories(input_file, output_file, min_lat, max_lat, min_lon, max_lon):
    valid_count = 0
    with open(input_file, 'r') as file, open(output_file, 'w', newline='') as output:
        reader = csv.reader(file)
        writer = csv.writer(output)

        header = next(reader)
        writer.writerow(header)  # Write the header to the output file

        current_trajectory = []
        current_tid = None
        valid_trajectory = True
        label_increment = 0
        tid_increment = 0

        for line in reader:

            # Foursquare has 7 columns, Geolife has 6
            if len(line) == 7:
                tid, label, lat, lon, day, hour, category = line
            else:
                tid, label, lat, lon, day, hour = line

            # Convert lat and lon to floats
            lat, lon = float(lat), float(lon)

            # Right at the start
            if current_tid is None:
                current_tid = tid

            # If we go to the next trajectory (this means the previous one is valid)
            if tid != current_tid:
                # A new trajectory has started
                if valid_trajectory :
                    for point in current_trajectory:
                        writer.writerow(point)
                    valid_count+=1
                    label_increment += 1
                    tid_increment += 1
                current_trajectory = []
                current_tid = tid
                valid_trajectory = True
                

            # Check if the point is outside the specified bounds
            if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
                valid_trajectory = False

            current_trajectory.append([tid_increment, tid_increment, lat, lon, day, hour])

        # Write the last valid trajectory
        if valid_trajectory :
            valid_count+=1
            for point in current_trajectory:
                writer.writerow(point)
    return valid_count



# using argparse to parse input and output file
parser = argparse.ArgumentParser()
parser.add_argument("input", help="The input file to restrict.")

parser.add_argument("output", help="The resultant restricted output.")

# Parse args
args = parser.parse_args()

if "geolife" in args.input.lower():
    # Geolife
    min_lat = 39.8279
    max_lat = 39.9877 # difference is ~0.16
    min_lon = 116.2676 
    max_lon = 116.4857 # difference is ~0.22
else:
    # Foursquare
    max_lat = 40.9883
    min_lat = 40.5508
    max_lon = -73.6857
    min_lon = -74.2696



# Call the function to filter and save trajectories within the specified bounds
valid = filter_and_save_trajectories(args.input, args.output, min_lat, max_lat, min_lon, max_lon)
print(f"{valid} Filtered trajectories saved successfully!")
