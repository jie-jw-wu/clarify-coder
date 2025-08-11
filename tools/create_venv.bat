@echo off
REM Set this to your Python (leave as "python" to use system)
set "PYTHON_EXE=C:/LocalData/Python310/python.exe"

echo Creating virtual environment...
"%PYTHON_EXE%" -m venv ../venv

call ../venv/Scripts/activate
echo Installing requirements...
pip install -r ../requirements.txt

echo [DONE] Setup complete.
