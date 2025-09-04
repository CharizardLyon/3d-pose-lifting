#Dataset

Change the .h5 files in the pose-2d for .csv and also leave the session at the start of the folder
To create the dataset use "extract_dataset.py" and change the paths, also the structure of the folder needs to be:

base: <br />
-session_person: <br />
--pose-2d: <br />
---camA.csv <br />
---camB.csv <br />
---camC.csv <br />
---camD.csv <br />
---camE.csv <br />
--pose-3d: <br />
---pose-3d.csv <br />
--raw-videos: <br />
---camA.mp4 <br />
---camB.mp4 <br />
---camC.mp4 <br />
---camD.mp4 <br />
---camE.mp4 <br />

#Training

in the train.py you can change the arguments for the training and to run it use the terminal
in the base path of the project and the run:

python -m scripts.train
