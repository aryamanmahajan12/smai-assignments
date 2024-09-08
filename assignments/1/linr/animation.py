import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from regression import PolynomialRegression
from PIL import Image, ImageDraw, ImageFont
import glob

# 1. Set up paths and parameters
image_folder = "images/"  # Folder to save images
output_gif_path = "animated_presentation.gif"
duration_per_frame = 1000  # milliseconds per frame

# Ensure the directory exists
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data['x'].values
y_train = train_data['y'].values
x_test = test_data['x'].values
y_test = test_data['y'].values

# Define degrees to use
degrees = [1, 2, 3, 4, 5]

# Function to calculate metrics manually
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    variance = np.var(y_pred)
    std_deviation = np.std(y_pred)
    return mse, variance, std_deviation

# 2. Calculate Metrics
metrics = {
    'MSE': [],
    'Variance': [],
    'Standard Deviation': []
}
for degree in degrees:
    model = PolynomialRegression(degree=degree)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse, variance, std_deviation = calculate_metrics(y_test, y_pred)
    metrics['MSE'].append(mse)
    metrics['Variance'].append(variance)
    metrics['Standard Deviation'].append(std_deviation)

# Save individual metric plots
def save_metric_plot(metric_values, metric_name, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, metric_values, marker='o')
    plt.xlabel('Degree')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} by Polynomial Degree')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

save_metric_plot(metrics['MSE'], 'Mean Squared Error (MSE)', os.path.join(image_folder, 'mse_plot.jpg'))
save_metric_plot(metrics['Variance'], 'Variance', os.path.join(image_folder, 'variance_plot.jpg'))
save_metric_plot(metrics['Standard Deviation'], 'Standard Deviation', os.path.join(image_folder, 'std_deviation_plot.jpg'))

# 3. Create Polynomial Fitting Images
def create_fitting_images(degrees, image_folder):
    image_files = []
    for i, degree in enumerate(degrees):
        model = PolynomialRegression(degree=degree)
        model.fit(x_train, y_train)

        plt.figure(figsize=(10, 6))
        plt.scatter(x_train, y_train, color='b', label='Training Data')

        x_curve = np.linspace(min(x_train), max(x_train), 500)
        y_curve = model.predict(x_curve)
        plt.plot(x_curve, y_curve, 'r-', label=f'Polynomial Fit (degree={degree})')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Polynomial Regression (degree={degree})')
        plt.legend()

        plot_filename = os.path.join(image_folder, f'gradual_fitting_{i}.jpg')
        plt.savefig(plot_filename)
        plt.close()
        image_files.append(plot_filename)
    
    return image_files

fitting_images = create_fitting_images(degrees, image_folder)

# 4. Create GIF with 2x2 Grid
def create_grid_image(images, grid_size, output_path):
    width, height = images[0].size
    grid_image = Image.new('RGB', (width * grid_size[0], height * grid_size[1]))

    for i, img in enumerate(images):
        x = (i % grid_size[0]) * width
        y = (i // grid_size[0]) * height
        grid_image.paste(img, (x, y))

    grid_image.save(output_path)

# Load images for the grid
mse_image = Image.open(os.path.join(image_folder, 'mse_plot.jpg')).resize((800, 600))
variance_image = Image.open(os.path.join(image_folder, 'variance_plot.jpg')).resize((800, 600))
std_deviation_image = Image.open(os.path.join(image_folder, 'std_deviation_plot.jpg')).resize((800, 600))

# Resize fitting images
fitting_images = [Image.open(p).resize((800, 600)) for p in fitting_images]

# Create frames for the GIF
grid_images = []

# Prepare 2x2 grid images for each polynomial degree
for i in range(len(degrees)):
    fitting_image = fitting_images[i % len(fitting_images)]
    grid_image_path = os.path.join(image_folder, f'grid_image_{i}.jpg')

    # Prepare the grid with specific metrics in cells
    images = [
        mse_image,          # Top-left
        variance_image,     # Top-right
        std_deviation_image, # Bottom-left
        fitting_image      # Bottom-right
    ]
    create_grid_image(images, (2, 2), grid_image_path)
    grid_images.append(grid_image_path)

# Create the GIF
frames = [Image.open(img_path) for img_path in grid_images]
frames[0].save(output_gif_path,
               save_all=True,
               append_images=frames[1:],
               duration=duration_per_frame,
               loop=0,
               optimize=True)

print("GIF created successfully!")
