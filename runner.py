import subprocess

for _ in range(0, 6):
    subprocess.call(['python', 'run_once.py'])
    print("Finished: running")
