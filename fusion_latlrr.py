import cv2
import numpy as np
from latent_lrr import latent_lrr  # Assuming latent_lrr is implemented as a Python function
import time

for index in range(1, 17):
    path1 = f'./source_images/IV_images/IR{index}.png'
    path2 = f'./source_images/IV_images/VIS{index}.png'
    fuse_path = f'./fused_images/fused{index}_latlrr.png'

    # Read images
    image1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale if necessary
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Normalize images to [0, 1]
    image1 = image1.astype(np.float64) / 255.0
    image2 = image2.astype(np.float64) / 255.0

    lambda_val = 0.8
    print("latlrr")

    # Process images using latent_lrr
    start_time = time.time()
    Z1, L1, E1 = latent_lrr(image1, lambda_val)
    Z2, L2, E2 = latent_lrr(image2, lambda_val)
    end_time = time.time()
    print(f"latlrr completed in {end_time - start_time:.2f} seconds")

    # Compute lrr and saliency components
    I_lrr1 = np.clip(np.dot(image1, Z1), 0, 1)
    I_saliency1 = np.clip(np.dot(L1, image1), 0, 1)
    I_e1 = E1

    I_lrr2 = np.clip(np.dot(image2, Z2), 0, 1)
    I_saliency2 = np.clip(np.dot(L2, image2), 0, 1)
    I_e2 = E2

    # Fusion
    F_lrr = (I_lrr1 + I_lrr2) / 2
    F_saliency = I_saliency1 + I_saliency2
    F = F_lrr + F_saliency

    # Display intermediate results
    cv2.imshow('I_saliency1', I_saliency1)
    cv2.imshow('I_saliency2', I_saliency2)
    cv2.imshow('F', F)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the fused image
    F_uint8 = (F * 255).astype(np.uint8)
    cv2.imwrite(fuse_path, F_uint8)
