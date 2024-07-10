import csv
import argparse

# Function to combine two CSV files with continuous tid and label values
def combine_csv_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w', newline='') as output:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(output)

        header = next(reader1)  # Read the header from the first file
        writer.writerow(header[:6])  # Write the header to the output file
        next(reader2)
        current_tid = None
        tid_increment = 1

        for line in reader1:
            if len(line) == 7:
                tid, label, lat, lon, day, hour, category = line
            else:
                tid, label, lat, lon, day, hour = line
            if current_tid is None:
                current_tid = tid
            if tid != current_tid:
                tid_increment += 1
                current_tid = tid
            writer.writerow([tid_increment, tid_increment, lat, lon, day, hour])
            
            # Update tid and label
            
        current_tid = None
        for line in reader2:
            if len(line) == 7:
                tid, label, lat, lon, day, hour, category = line
            else:
                tid, label, lat, lon, day, hour = line
            if current_tid is None:
                current_tid = tid
            if tid != current_tid:
                tid_increment += 1
                current_tid = tid
            writer.writerow([tid_increment, tid_increment, lat, lon, day, hour])
            
            # Update tid and label
            
            

# Getting the input file paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("file1", help="First file to be combined.")
parser.add_argument("file2", help="Second file to be combined.")


# And the output file path
parser.add_argument("output_file", help="Output file path.")

# Parse args
args = parser.parse_args()

combine_csv_files(args.file1, args.file2, args.output_file)
print("CSV files combined successfully!")
