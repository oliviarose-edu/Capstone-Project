# Import libraries
import tkinter as tk                                #used for gui outputs
import json                                         #used for reading json files
import numpy as np                                  #used for numerical operations
from sklearn.ensemble import RandomForestClassifier #used for training
from PIL import Image, ImageTk                      #used for reading images
import random                                       #used to select at random

# Load the dataset
with open('archive/shipsnet.json') as dataFile: #just done in the same directory
    data = json.load(dataFile)

X = np.array(data['data'])   #pixel values
y = np.array(data['labels']) #labels

# Create the RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, random_state = 42) #80% training set, 20% testing set
clf.fit(X, y) #fit it into the classifier

# Detect ships
def detectShip(imageData):
    result = clf.predict([imageData])[0] #predict with image data
    if result == 1: #if ship detected
        result_lbl.config(text = "Result: Ship Detected") #print
    else: #if ship NOT detected
        result_lbl.config(text = "Result: No Ship") #print

# Load a random image
def loadImage():
    imageIndex = random.randint(0, len(X)-1) #select random number from number of images
    imageData = X[imageIndex] #load image that corresponds to the random index selected
    
    image = Image.fromarray(imageData.reshape((80, 80, 3)).astype('uint8')) #reshape the data so it can be analysed and displayed
    image = ImageTk.PhotoImage(image) #convert into photo image so it can be displayed as an iamge
    canvas.create_image(0, 0, anchor = tk.NW, image = image) #just sorting out placement here
    canvas.image = image
    detectShip(imageData) #perform the detect ship function on this image

# Create root
root = tk.Tk()
root.title("Ship Detector")

# Make canvas to display image
canvas = tk.Canvas(root, width = 100, height = 100)
canvas.grid(row = 0, column = 0, columnspan = 2) #placement

# Create a button to load a random image
load_btn = tk.Button(root, text = "Load image", command = loadImage)
load_btn.grid(row = 1, column = 0, sticky = 'w') #placement

# Label to display the detection result
result_lbl = tk.Label(root, text = "")
result_lbl.grid(row=1, column=1, sticky='e') #placement

root.mainloop()
