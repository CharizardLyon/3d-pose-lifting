Dataset

To create the dataset use "extract_dataset.py" and change the paths, also the strucuture of the folder needs to be:
Change the .h5 files in the pose-2d for .csv

base:
-person:
--pose-2d:
---camA.csv
---camB.csv
---camC.csv
---camD.csv
---camE.csv
--pose-3d:
---pose-3d.csv
--raw-videos:
---camA.mp4
---camB.mp4
---camC.mp4
---camD.mp4
---camE.mp4

in the train.py you can change the arguments for the training and to run it use the terminal
in the base path of the project and the run:

python -m scripts.train
