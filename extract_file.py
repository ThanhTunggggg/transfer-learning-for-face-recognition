import argparse
import base64
import csv
import os
# import magic # Detect image type from buffer contents (disabled, all are jpg)

parser = argparse.ArgumentParser()
parser.add_argument('croppedTSV', type=str)
parser.add_argument('--outputDir', type=str, default='raw')
args = parser.parse_args()

with open(args.croppedTSV, 'r') as tsvF:
    reader = csv.reader(tsvF, delimiter='\t')
    i = 0
    for row in reader:
        MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

        saveDir = os.path.join(args.outputDir, MID)
        savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

        # assert(magic.from_buffer(data) == 'JPEG image data, JFIF standard 1.01')

        os.makedirs(saveDir, exist_ok=True)
        with open(savePath, 'wb') as f:
            f.write(data)

        i += 1

        if i % 1000 == 0:
print("Extracted {} images.".format(i))