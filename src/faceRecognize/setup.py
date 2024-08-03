import sys
from cx_Freeze import setup, Executable

setup(
    name="Face Recognition",
    version="1.0",
    description="Your application description",
    executables=[Executable("app.py", base=None)]
)
