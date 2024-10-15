import numpy as np
import cv2


aug_list = ['_pseudo_dsms_automasking_noadjust']
seq_list = ['sigmoid_t3_b', 'trans_t4_a', 'trans_t2_b', 'trans_t3_a', 'trans_t4_a', 'trans_t4_b']
idx_list = ['0000', '0160', '0007', '0000', '0000', '0000']

model_list =  ['IID'] #['monodepth2', 'monovit', 'IID'] #

for model in model_list:
    rows =[]
    for seq, idx in zip(seq_list, idx_list):
            
        # Load image paths
        reflec_files =[f"/raid/rema/outputs/undisttrain/undist/{model}/finetuned_mono_hkfull_288{aug}/models/weights_19/{seq}/decomposed/reflect{idx}_color.png" for aug in aug_list]
        light_files = [f"/raid/rema/outputs/undisttrain/undist/{model}/finetuned_mono_hkfull_288{aug}/models/weights_19/{seq}/decomposed/light{idx}_color.png" for aug in aug_list]
        image_files = [f"/raid/rema/data/C3VD/Undistorted/Dataset/{seq}/{idx}_color.png"]
        annot_files = [f"/raid/rema/data/C3VD/Undistorted/Annotations_Dilated/{seq}/{idx}_color.png"]
        inpainted_files = [f"/raid/rema/data/C3VD/Undistorted/Inpainted_HKgen9/{seq}/{idx}_color.png"]
        
        combined_hor = []
        for original_file, light_file, reflec_file, annot_file, inpainted_file in zip(image_files, light_files, reflec_files, annot_files, inpainted_files):
            original = cv2.imread(original_file, cv2.IMREAD_UNCHANGED)
            light = cv2.imread(light_file, cv2.IMREAD_UNCHANGED)
            reflec = cv2.imread(reflec_file, cv2.IMREAD_UNCHANGED)
            annot = cv2.imread(annot_file, cv2.IMREAD_UNCHANGED)
            inpainted = cv2.imread(inpainted_file, cv2.IMREAD_UNCHANGED)
            
            # Convert grayscale light image to RGB
            if len(light.shape) == 2:  # Check if the image is grayscale
                light = cv2.cvtColor(light, cv2.COLOR_GRAY2RGB)

            # Ensure both images are in float format for multiplication
            light = light.astype(np.float32) / 255.0
            reflec = reflec.astype(np.float32) / 255.0

            # Get reconstructed image from light and reflectance
            reconstructed = light * reflec

            # Convert the reconstructed image back to 8-bit format
            reconstructed = (reconstructed * 255).astype(np.uint8)

            # Calculate the absolute difference between the original and reconstructed images
            difference = cv2.absdiff(original, reconstructed)

            # Convert the difference to grayscale
            difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

            # Apply a threshold to highlight big changes
            _, thresholded = cv2.threshold(difference_gray, 50, 255, cv2.THRESH_BINARY)
            
            # make thresholded rgb
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)


            combined_hor.append(original)
            combined_hor.append(inpainted)
            combined_hor.append(reconstructed)
            combined_hor.append(annot)
            combined_hor.append(thresholded)

            
            combined_horizontal = np.hstack(combined_hor)
        
        rows.append(combined_horizontal)
    
    # Stack the rows vertically
    result = np.vstack(rows)
    
    # Save the result
    cv2.imwrite(f'/raid/rema/outputs/undisttrain/undist/visualresultsspec_reconst{model}C3VD{str(aug_list)}.png', result)