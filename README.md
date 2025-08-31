# MNIST Digit Classifier

![Image of the application](images/demo.png)

## What is this?
This is a digit classifier written from scratch by me to practise using c++ and using neural 
networks.
This project uses a neural network written from scratch trained on the MNIST image dataset 
to detect what digit was drawn.<br>
This is based on roughly Samson Zhang's neural network 
tutorial, however this version is completely rewritten in c++, and has an interactive gui, and is slightly more 
optimised.


## How can I try this out?
Executable files are available in the Releases section on the right hand side of this
page, click the latest release and download the correct executable file.
click the 'run' file inside the folder. <br>
There are also instructions to build this project yourself.

## How do I use this?
Click the 'train' button to train the neural network. The network has one hidden layer 
with 10 neurons. <br>
Leave the model to train, when a suitable accuracy you are satisfied with is attained, click stop
(this may take some time). <br>
Draw a digit in the empty space, the model will predict what it is.
At the moment, there is no pre-trained model available, and you must train the neural network yourself.

## Build Project
If a binary isn't available, and you are comfortable using the terminal, follow the next steps to 
build the project from scratch using cmake.

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
Run cmake
<pre>
cmake --build .
</pre>
Run the executable
<pre>
./run
</pre>

