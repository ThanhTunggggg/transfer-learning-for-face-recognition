# Transfer learning for face recognition

## Datasets
- Download [the shorten MS-Celeb dataset (around 4GB)](https://drive.google.com/open?id=1I24at7mUzo1R3jU8HNbT7bXOOpU6kx7C).
- Clean image list in data_clean.txt
## Model
- Transfer learning [pretrained model LCNN29](https://www.dropbox.com/s/yn66p77w7estfga/model.zip?dl=0).
- Training final layer with 5031 output units  for 5031 people in the shorten MS-Celeb dataset.
## Steps
- Run: ./extract_file.py <tsv_file> --outputDir <directory_to_extract_to>
- Run img2h5df.py to convert image data to h5 format.
- Run LCNN29.py to training or run LCNN29_transfer.py to transfer leaning.
## Referencs
- [Model for entire MS-Celeb dataset](https://github.com/yxu0611/Tensorflow-implementation-of-LCNN).
