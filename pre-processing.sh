echo "Converting PLT files to CSV" 
python3 pre-processing/plt-csv.py 
echo "Restricting trajectories to domain"
python3 pre-processing/restrict-trajectories.py geolife_data.csv data/geolife/restricted_geolife.csv