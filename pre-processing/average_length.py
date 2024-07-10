import csv

# Read the file and calculate the average trajectory length
file_path = "combined_forsquare.csv"  # Replace with the path to your file
trajectory_lengths = {}

with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row

    for row in reader:
        tid, label, lat, lon, day, hour = map(float, row[0:6])

        # Create a dictionary to store trajectory lengths for each TID
        if tid not in trajectory_lengths:
            trajectory_lengths[tid] = 0

        # Increment the trajectory length for the current TID
        trajectory_lengths[tid] += 1

# Calculate the average trajectory length for each TID
average_lengths = {}
for tid, length in trajectory_lengths.items():
    average_lengths[tid] = length

# Calculate the overall average trajectory length
total_length = sum(average_lengths.values())
total_trajectories = len(average_lengths)

if total_trajectories > 0:
    average_length = total_length / total_trajectories
    print(f"Average Length of Location Trajectories: {average_length}")

else:
    print("No trajectories found in the file.")
