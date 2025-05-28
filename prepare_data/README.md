# Dataset Preparation Instructions
To prepare the datasets for training and testing, follow the steps below:

## C3VD Dataset Preparation
Download the C3VD dataset from [here](https://durrlab.github.io/C3VD/). Place them in `Datasets/C3VD/Dataset/`

- We only need color and depth images, so you can skip the other files.

- Use [prepare_data/cap_frame_926.sh](prepare_data/cap_frame_926.sh) to cap the frames to 926 frames per video.


## Hyper Kvasir BBPS-2-3 Dataset Preparation
Download the Hyper Kvasir BBPS-2-3 dataset from [here](https://osf.io/mh9sj/files/osfstorage#) (`OSF Storage/labeled-videos/lower-gi-tract/quality-of-mucosal-view/BBPS-2-3/`). Place them in `Datasets/BBPS-2-3/Frames/`

- Run the following scripts to prepare the BBPS-2-3 dataset:

    ```bash
    bash prepare_data/video2Frames.sh
    bash prepare_data/resizeFrames288.sh
    ```
- Use [prepare_data/cap_frame_926.sh](prepare_data/cap_frame_926.sh) to cap the frames to 926 frames per video.


## Undistortion and Inpainting
The images in the C3VD and BBPS-2-3 datasets are distorted. To use them for training, they need to be undistorted.

- You can use the [prepare_data/undistort.py](prepare_data/undistort.py) script to undistort the images.

- For convenience, the undistorted masks and inpainted images are available for download [here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/EkdKZv1ZRJ9KgiQBGeFxdFEB4Dyd-F1JNTKAGbyIRN_hHA?e=sw9K6v)

> **_NOTE:_**
>
> To regererate the undistorted masks and inpainted images:
> - Follow the tutorial in [Endo-STTN](https://github.com/endomapper/Endo-STTN.git) to generate the distorted versions.
> - Adapt [prepare_data/undistort.py](prepare_data/undistort.py) to undistort the generated masks and inpainted images.




## Directory Structure
The folders should be structured as follows:

```
C3VD/
    Undistorted/
        Dataset/
            cecum_t1_a/
                0000_color.png
                0000_depth.tiff
                0001_color.png
                0001_depth.tiff
                ...
            ...
        Inpainted_HKgen9/
            cecum_t1_a/
                0000_color.png
                0001_color.png
                ...
            ...     
        Annotations_Dilated/
            cecum_t1_a/
                0000_color.png
                0001_color.png
                ...
            ...
```

```
BBPS-2-3/
    Undistorted/
        Frames/
            1c1ca45a-315e-4ad0-9343-2425d2faf648/
                0000.png
                0001.png
                ...
            ...
        Inpainted_gen9/
            1c1ca45a-315e-4ad0-9343-2425d2faf648/
                0000.png
                0001.png
                ...
            ...  
        Annotations/
            1c1ca45a-315e-4ad0-9343-2425d2faf648/
                0000.png
                0001.png
                ...
            ...  
```
