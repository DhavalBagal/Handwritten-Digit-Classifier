# Handwritten Digit Classifier

The project entails a web application which allows you to classify handwritten digits.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites

You need to have the following packages installed on your device before running the project.

	python==3.6 (or above)
	torch==1.6.0
	flask==1.1.1
	torchvision==0.7.0
	
### Running the project

1. Clone the repository to your local system.
2. Change your default directory to the directory where you cloned the project using ***cd*** command.
3. Open your terminal and execute the Server. py script
		
		python3 Server.py

4. Go to your browser and enter the following:
	
		http://localhost:8000/
	
## Directory Structure and Breakdown

**`Server.py`** - Server side script that interacts with the web application

**`Model.py`** - CNN Model for image classification

**`Train.py`** - Trainer class to train the CNN model further

**`model.pth`** - Saved model file

**`templates/index.html`** - Web application

**`static/`** - JS classes, font file and images

**`NN-from-scratch-code/Helper.py`** - Helper classes for implementing a Neural Network from scratch

**`NN-from-scratch-code/Trainer.py`** - Trainer class to train the CNN model further

**`NN-from-scratch-code/DigitRecognizer.py`** - DigitRecognizer class to classify the input

**`NN-from-scratch-code/model.json`** - Model saved in JSON format

## Built with

**React** - Web framework for Web application
**Flask** - Server Side Backend
**Pytorch** - Building the CNN model
