# Latitude | Longitude | Random Number (why not ig) | Altitude | DatetimeNumerical | Date String | Time String
import os
from datetime import datetime

# folder path



tid = 0
tid1 = 0
with open("geolife_data.csv", "w+") as csv1:
    csv1.write("tid,label,lat,lon,day,hour\n")
    for folder in range(1, 181):
        dir_path = f"Geolife Trajectories 1.3/Data/{str(folder).zfill(3)}/Trajectory"
        numfiles = 0
        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                numfiles += 1

        # First 6 lines are useless

        files = os.listdir(dir_path)
        for filename in files[: int(len(files))]:
            file = open(dir_path + f"/{filename}", "r")
            for i in range(6):
                file.readline()
            numLines = 0
            lines = []
            for line in file:
                vars = line.split(",")
                this = datetime.strptime(vars[5], "%Y-%m-%d")
                time = vars[6].split(":")
                lines.append(
                    [
                        str(tid1),
                        str(tid1),
                        vars[0],
                        vars[1],
                        str(this.weekday()),
                        str(int(time[0])),
                    ]
                )
                numLines += 1
                if numLines >= 144:
                    break
            if numLines <= 96:
                continue
            for line in lines:
                csv1.write(",".join(line) + "\n")
            tid1 += 1
print(f"Done converting {tid1} trajectories.\nSaved in geolife_data.csv!")