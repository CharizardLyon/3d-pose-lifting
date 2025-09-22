# 📂 Dataset & Training Guide

This repository is organized to work with multi-camera (2D & 3D) data and train pose estimation models.  

---

## 📊 Dataset

1. **Data format**  
   - Replace `.h5` files inside the `pose-2d/` folder with `.csv` files.  
   - Each session should be placed at the root level of the base folder.  

2. **Required folder structure**  

```bash
base/
└── session_person/
├── pose-2d/
│ ├── camA.csv
│ ├── camB.csv
│ ├── camC.csv
│ ├── camD.csv
│ └── camE.csv
│
├── pose-3d/
│ └── pose-3d.csv
│
└── raw-videos/
├── camA.mp4
├── camB.mp4
├── camC.mp4
├── camD.mp4
└── camE.mp4
```

3. **Dataset creation**  
   - Use the script [`extract_dataset.py`](extract_dataset.py).  

---

## 🏋️ Training

1. **Run training**  
   From the project’s base directory, execute:  

   ```bash
   python -m scripts.train
   ```

##  Unity Integration

To visualize the 3D hand pose in Unity, follow these steps:

1. Scene setup.
   * Create an empty object called “Handler.”
   * Create a folder called “HAND.”

This object will contain:

* The **21 points** (representing hand joints).

* The **lines** that connect those joints.

Each line is associated with the script `MakingLines.cs`. Once this script has been added to the Line 0-1 element (for example), the origin point (0) and destination point (1) must be set.

The structure should be:
```bash

Handler # Empty game object
HAND/
├── Points/
│     ├── Point(0)
│     ├── Point(1)
│     └── ... Point(20)
│
├── Lines/
│     ├── Line(0-1)
│     ├── Line(1-2)
│     └── ... Line(19-20)
```
>⚡ The connections between points are based on the MediaPipe hand reference diagram.

**2. Networking setup**

* The IP and Port of "Handler" must be configured to receive incoming 3D joint data packets.

**3. Attach the script**

* Assign the script `HandPoseRig.cs` to the manejador object.

**4. Running the animation**

* Press Play in Unity. The scene will start listening for incoming data.

* Run the Python script `inference_sockets_realtime.py` to send the real-time 3D coordinates obtained from the trained model.

**This setup allows Unity to render a live animation of the hand using the predicted joint coordinates.**
