#!/usr/bin/env python3
"""
Startup script for the ML API service
"""
import subprocess
import sys
import os

def main():
    # Change to the ml/api directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Start the FastAPI server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8544",
        "--reload"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()