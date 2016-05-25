import subprocess
import sys

def set_fan_speed(speed):
    cmd1 = ['nvidia-settings', '-a', '[gpu:0]/GPUFanControlState=1']
    cmd2 = ['nvidia-settings', '-a', '[fan:0]/GPUTargetFanSpeed={}'.format(speed)]
    subprocess.run(cmd1)
    subprocess.run(cmd2)
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        speed = sys.argv[1]
    else:
        speed = 30
    set_fan_speed(speed)