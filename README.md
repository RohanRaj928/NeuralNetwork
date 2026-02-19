# MNIST Digit Classifier

![Image of the application](images/demo.png)

## What is this?
This is a numerical digit classifier written entirely from scratch using C++ to experiment with neural networks.  
This project uses a neural network trained on the MNIST image dataset to classify numerical digits. The code is roughly based on Samason Zhang's neural network tutorial, however this project is written in C++ instead of python and includes an interactive gui.

## How can I try this out?
You must clone and build this project yourself (Instructions below for UNIX) <br>
I will include a .exe file for windows soon, or you can build this project yourself with cmake

Clone this repository
<pre>
git clone https://github.com/RohanRaj928/NeuralNetwork
cd NeuralNetwork
</pre>
Create a build folder
<pre>
mkdir build
cd build
</pre>
Run cmake (Must be installed)
<pre>
cmake ..
make 
</pre>
Run the executable by opening the folder and clicking the executable, or running on console


## How do I use this?
Open the application (This may take a while) <br>
Click the 'train' button to train the neural network. <br>
Leave the model to train, and click 'Stop' when finished (This may take a while) <br>
Draw a digit in the empty space, the model will make a prediction

