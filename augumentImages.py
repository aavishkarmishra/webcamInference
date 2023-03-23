import os
from PIL import Image, ImageOps, ImageEnhance
import random

# Define a function to apply random augmentations to an image
def augment_image(image):
    # Randomly flip the image horizontally
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    
    # Randomly rotate the image
    rotate_angle = random.randint(-10, 10)
    image = image.rotate(rotate_angle)
    
    # Randomly adjust the brightness and contrast of the image
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    
    # Convert the image to black and white
    image_greyscale = ImageOps.grayscale(image)
    
    # Invert the image to create inverted greyscale
    image_inverted = ImageOps.invert(image_greyscale)
    
    # Return the augmented images
    return image_greyscale, image_inverted

# Create a new directory for the augmented images
if not os.path.exists('augmented'):
    os.makedirs('augmented')

# Loop through all the files in the current directory
for filename in os.listdir('.'):
    # Check if the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image and apply random augmentations
        image = Image.open(filename)
        augmented_greyscale, augmented_inverted = augment_image(image)
        
        # Save the augmented images in the "augmented" directory
        output_greyscale_path = os.path.join('augmented', filename.split('.')[0] + '.png')
        augmented_greyscale.save(output_greyscale_path)
        #output_inverted_path = os.path.join('augmented', filename.split('.')[0] + '_inverted.png')
        #augmented_inverted.save(output_inverted_path)
