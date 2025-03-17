#!/bin/bash
# This script runs the improved Titanic model with the correct Python installation

# Attempt to deactivate any virtual environment
deactivate 2>/dev/null || true

# Use the Python from 3.10 framework that has our packages installed
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 titanic_improved.py

# If that fails, try the 3.12 framework
if [ $? -ne 0 ]; then
  echo "Trying Python 3.12..."
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 titanic_improved.py
fi

# If that also fails, try the system Python
if [ $? -ne 0 ]; then
  echo "Trying system Python..."
  /usr/bin/python3 titanic_improved.py
fi 