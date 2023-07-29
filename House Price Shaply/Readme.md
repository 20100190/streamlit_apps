-
# My Project

This is a simple Python ipynb project that demonstrates XYZ.


## Kernel in Ipynb
1. Ctrl + Shift + P will open a dialogue to select interperator.
2. Type there Python: Select interperator
3. click on + add path if desired interperator is not there already. Path is till bin/python3.9 from your venv folder.
4.And then from right upper corener you can select desired kernel
5. And to activative that interperator in cmd we use `venv\Scripts\activate.bat`

## How to run
1. `ipython notebook_name.ipynb` will run like python file and ask for user input as well
2. `jupyter nbconvert --to html notebook_name.ipynb` will render html file in same directory cannot get user input
3. 

## Installation

0. To create virtual environment we use `python -m venv venv` one time only
1. Clone the repository.
2. Activate the virtual environment:
    - For Windows: `venv\Scripts\activate.bat`
    - For macOS/Linux: `source venv/bin/activate`
3. Install the project dependencies:
    ```bash
    pip install -r requirements.txt
    pip freeze -> requirements.txt
    ```
4. To deactivate virtual env just use `deactivate`


## Usage

1. Run the project using the following command:
   ```bash
   python main.py
   ```


## Cmd Tips

1. To create a file we use `touch > filename.ext`, `echo > filename.ext` print file content in cmd console
2. To create a folder we use `mkdir foldername`
3. To change directory we use cd Path
4. To Open a file through notepad or any text editor we use `notepad filename.ext`
5. cmd has inbuilt nano editor where ^ is control and to save we use `Ctrl + X`
6. rmdir / del to remove directory/folder and file respectively.



"# First-app" 
