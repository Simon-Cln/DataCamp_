# BreakThrough_
**Overview**
- This repository contains the codebase for an eye cancer image classification project. The main objective is to develop a tool capable of detecting cancer in eyes using a dataset of eye images. The user can upload an image of an eye through a web interface and receive feedback regarding the likelihood of that eye having cancer. This solution will soon integrate cloud services for image preprocessing and classification. 

**Structure**
The repository is organized as follows:

- .github/workflows: Contains workflows for Azure App deployment.
eyes_be and eyes_fe: Backend and frontend directories, respectively.

- src: Contains the source code for the frontend application.
- Other root level files: Include configuration files for git and package management.

## Setup and Installation
- Prerequisites:
Make sure you have Python and Node.js installed on your machine.
Some functionalities rely on specific Python packages and npm modules. Please refer to requirements.txt and package.json for the list of dependencies.
Steps:
- Clone the Repository:

**git clone https://github.com/Simon-Cln/DataCamp**
Navigate to the Project Directory:
cd [Your Repo Name]


## **Backend Setup:**


- Navigate to the backend directory (eyes_be):
cd eyes_be
Install the required Python packages:
pip install -r requirements.txt


## **Frontend Setup:**


- Navigate to the frontend directory (eyes_fe):
cd eyes_fe
Install the necessary npm modules:

npm install
## **Model Setup:**

**The repository does not contain the h5 model file. You will need to generate this by running test_datacamp.py:**
- **python test_datacamp.py**

- Then you can type **python app.py** in the back end folder and **npm run start** in the front end one.

**Important Notes:**
The Excel files and PNG images are not included in this repository. However, this shouldn't be a concern as the model is already trained.

**Contributing**
If you wish to contribute, please submit a pull request, and it will be reviewed by the maintainers.

# Getting Started with Create React App

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!


### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify


