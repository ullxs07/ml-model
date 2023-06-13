#import cv2
#
#def preprocess_image(image):
#    #Convert to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    #Resize the image
# resized_image = cv2.resize(gray_image, ( 124, 124))
#
#    #Apply CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 #enhanced_image = clahe.apply(resized_image)
#
 #return enhanced_image

#import cv2
#import numpy as np

#def preprocess_image(image):
    #if len(image.shape) == 2:  # Grayscale image
        #gray_image = image
   # else:  # Color image
       # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #  print("Grayscale image shape:", gray_image.shape)
  #  resized_image = cv2.resize(gray_image, (124, 124))
  #  processed_image = resized_image.reshape((1, 124, 124, 1))
  # 3 processed_image = processed_image.astype('float32') / 255.0

    #return processed_image

#import cv2

#def preprocess_image(image):
    # Convert to grayscale and resize the image
  #  enhanced_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (124, 124))
#
    # Apply CLAHE
  #  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  #  enhanced_image = clahe.apply(enhanced_image)
#
   # return enhanced_image

import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image to (124, 124)
    resized_image = cv2.resize(image, (150, 150))
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    
    # Expand dimensions to match (1, 124, 124, 3)
    preprocessed_image = np.expand_dims(clahe_image, axis=-1)
    preprocessed_image = np.repeat(preprocessed_image, 3, axis=-1)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    return preprocessed_image

# Example usage
#image_path = 'path/to/your/image.jpg'
#preprocessed_image = preprocess_image(image_path)
#print(preprocessed_image.shape)  # Output: (1, 124, 124, 3)#