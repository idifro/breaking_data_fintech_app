#!/usr/bin/env python3
"""
Startup script to run the Streamlit dashboard
Just run this file to start the web app
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the dashboard web app"""
    
    print(" BDF Stock Market Dashboard - Streamlit Startup")
    print("=" * 50)
    
    # Figure out where the dashboard files are
    dashboard_dir = Path(__file__).parent
    
    # Make sure app.py exists before trying to run it
    app_file = dashboard_dir / "app.py"
    if not app_file.exists():
        print(f" Dashboard app file not found: {app_file}")
        return 1
    
    print(f"Dashboard location: {dashboard_dir}")
    print(f"Starting Streamlit app: {app_file}")
    
    # Change to dashboard directory so paths work correctly
    os.chdir(dashboard_dir)
    
    try:
        # Launch streamlit with proper settings
        print("\nStarting Streamlit dashboard...")
        print("Dashboard will open at: http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        
        # Use current Python interpreter to run streamlit
        python_exe = sys.executable
        subprocess.run([
            python_exe, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501", 
            "--server.headless", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        # User pressed Ctrl+C to stop
        print("\nDashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f" Failed to start dashboard: {e}")
        return 1
    except Exception as e:
        print(f" Unexpected error: {e}")
        return 1
    
    print(" Dashboard stopped successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())