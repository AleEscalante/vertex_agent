import os
import sys
import subprocess

def main():
    """Run the Django development server"""
    # Install requirements
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run migrations
    print("Running migrations...")
    subprocess.check_call([sys.executable, "manage.py", "makemigrations"])
    subprocess.check_call([sys.executable, "manage.py", "migrate"])
    
    # Start server
    print("Starting server...")
    subprocess.check_call([sys.executable, "manage.py", "runserver", "0.0.0.0:8000"])

if __name__ == "__main__":
    main()
