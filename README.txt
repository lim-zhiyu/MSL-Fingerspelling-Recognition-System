Name: Lim Zhi Yu
Student ID: 0347568


- About -
This folder contains the project files used to create the real-time Malaysian Sign Language (MSL) recognition application as part of my Capstone Project.
Below are descriptions of some of the files and folders in the project directory:
    [file] project_notebook.ipynb - The Jupyter Notebook used to create the model
    [file] actions.h5 - The trained model
    [folder] Logs - The model training log recorded using TensorBoard
    [folder] MP_Data - Contains the keypoint data used to train and test the model
    [file] app.py - The prototype web app of the MSL recognition system
    [folder] templates - Contains the HTML file of the web app
    [folder] static - Contains the images and stylesheet used by the web app
    [file] environment.yml - The project environment file exported from Anaconda Navigator


- Instruction -
The real-time MSL recognition system can be run as the web app or in the Jupyter notebook. It is recommended to run the system as the web app because it contains a reference section for the MSL signs. However, if the processing performance of the system is poor in the web app (very low framerate), running the system in the Jupyter notebook may improve the processing performance of the system.

To run the system in the notebook, open project_notebook.ipynb in Jupyter Notebook and follow the instruction in the first block.

To run the system as a web app, follow the instuction below:
1. Open Anaconda Navigator, import environment.yml and switch to the imported environment named "zhiyu-capstone".
2. Open Powershell Prompt in Anaconda Navigator and navigate to the project directory (where app.py is located).
3. Run the command "Python .\app.py" (without quotes) and visit the url in the output with a browser.
4. To stop the webcam and close the web app, press Ctrl + Pause Break in Powershell Prompt or close Powershell Prompt.


- Tips for Improving Sign Recognition -
Various functions have been implemented to help improve the experience of getting a sign recognized and testings have been made to ensure every sign can be recognized by the model. However, if you find it difficult to get a sign recognized, try the tips below:
    - Follow the sign as closely as possible in the reference images.
    - Adjust your arm's angle so it is straight vertically (or horizontally, depending on the sign).
    - Adjust your hand's distance to the camera.
    - Adjust your hand's tilt, swivel, and pivot.
    - Position your hand at the centre of the webcam view.
    - For dynamic signs like 'J', move your hand slower to maintain the action for a longer duration.
    - Use the output controls to manually append the first or second actions if the action does not get appended automatically because its probability is not high enough or it does not stay high for long enough.


- Troubleshooting -
    - If Anaconda Navigator fails to quit, end the Python process in Task Manager.
    - If the webcam does not open, make sure no other program is using the webcam and the webcam is switched on.
    - If the webcam still does not open or a wrong camera is used (if your computer has multiple cameras), you may need to modify the value in the argument of the cv2.VideoCapture() function in app.py or project_notebook.ipynb to a value that represents the camera on your computer that you want to use (default value is 0).
    - Additional troubleshooting situations are included in app.py and project_notebook.ipynb.