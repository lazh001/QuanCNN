import os
import numpy as np
from torchvision import datasets
from PIL import Image

# 此代码用于从 MNIST 数据集中提取图像并保存为 PNG 文件。
# 使用 PyTorch 的 torchvision 库加载 MNIST 数据集。
# 每个图像将按照标签存储在对应的子文件夹中。

# Define the output directory for saving PNG images
output_dir = "./mnist_npy"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it does not exist

# Load the MNIST dataset
# Download=True ensures that the dataset is downloaded if not already present
mnist_data = datasets.MNIST(root="../dataset", download=False)

# Iterate over the dataset and save each image as a PNG file
for idx, (image, label) in enumerate(mnist_data):
    # Print progress for every 1000 images
    if (idx + 1) % 1000 == 0:
        # Convert the image (PIL format) to a numpy array
        image_array = np.array(image)

        # Define the directory for the label (e.g., "mnist_png/0/" for label 0)
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)  # Create the directory for the label if it does not exist

        # Define the path to save the PNG file
        file_path = os.path.join(label_dir, f"{idx}.npy")
        # Save the image
        #image.save(file_path, "PNG")
        np.save(file_path, image_array)
        #np.savetxt(file_path, image_array, fmt='%d', delimiter=',')
        print(f"Saved {idx + 1}/{len(mnist_data)} images.")

print("All images have been saved as PNG files.")
