# Instructions for Film Lab

This lab can be done with the software of your choice, like Film QA Pro which you have access to on CAVIPC00037 (if you stop by BC Cancer, this is in the room next to the students lab, where the film scanner is attached to.). 

You should be able to remote in using citrix, but you need to know your BC Cancer ID and password. You can then go onto apps.phsa.ca, sign in, and follow [this guide](https://webassets.phsa.ca/citrix/User%20Guide%20for%20Remote%20Network%20Access.pdf) on how to get that set up.

Alternatively (and likely easier), you can use this python program that Dr. Clay Lindsay originally set up. 

However, to use this properly, you must read and follow this procedure:

1. Use python 3.10 or above. I (Dr. Kyle Bromma) personally used 3.10.2. If something doesn't work, try changing python version. 
    - Check version with ```python --version``` in your cmd prompt
    - You might need to use ```python3``` for all commands instead of ```python```

2. Make a virtual environment
    - If using windows: ```python -m venv .venv```
    - If using bash, need to install ```python3.10-venv``` (if using python 3.10, adjust for your version), then you can use ```python -m venv .venv```
    - Further, to run, please install the following if you use ubuntu (you'll have to find alternatives for other platforms):
    ```
    apt install -y libgl1
    apt install python3-tk
    ```
3. Enter your virtual environment and install the requirements file
    - If in windows: ```.\.venv\Scripts\activate```
    - If in bash: ```source .venv/bin/active```
    - For all: ```pip install -r requirements.txt```
4. You can run the program now using ```python main.py```. However, the data it is using is sample data. You will have to use the files provided to you to analyze the film you exposed during the lab. This includes the calibration files and the actual test.
    - *DO NOT* submit the sample_data as your data.
    - You can save the new data in a new folder and edit the code to read from there.
    - You will also likely need to change the ROI where the calibration is read from on the film. Instructions are in the code comments.