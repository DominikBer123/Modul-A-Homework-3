import subprocess
import sys

def main():
    homework_files = [
        "Large_Shiffle_False.py",
        "Large_Shiffle_True.py",
        "Small_Shiffle_False.py",
        "Small_Shiffle_True copy.py"
    ]
    
    print(f"Starting execution of {len(homework_files)} files...")
    
    for script in homework_files:
        print(f"Running: {script}")
        try:
            # sys.executable ensures it uses the same Python environment you are currently running
            subprocess.run([sys.executable, script], check=True)
            print(f"Completed: {script}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
            print("Stopping execution loop.")
            break

if __name__ == "__main__":
    main()