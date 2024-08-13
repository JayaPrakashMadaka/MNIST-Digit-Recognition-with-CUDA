import cv2
import numpy as np
import os

np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

def process_image(img_path, output_dir):
    img = cv2.imread(img_path, 0)
    if img is None:
        print("Unable to read image:", img_path)
        return
    
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    
    img = img.reshape(28, 28, -1)
    img = img / 255.0
    output_file = os.path.join(output_dir, os.path.basename(img_path).replace('.png', '.txt'))
    with open(output_file, "w") as file:
        for i in range(28):
            for j in range(28):
                file.write("{}\n".format(img[i][j][0]))


input_dir = "img"
output_dir = "pre-proc-img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = os.listdir(input_dir)

# Process the first 1000 images
ctr = 0
for i, file in enumerate(files):
    ctr += 1
    img_path = os.path.join(input_dir, file)
    process_image(img_path, output_dir)
print(ctr)