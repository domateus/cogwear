import subprocess

for _ in range(0, 3):
    subprocess.call(['python', 'run_once.py'])
    print("Finished: running")
