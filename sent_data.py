# the program is just use to send the song information
import serial
import time
waitTime = 0.1
songset = [
[262, 262, 392, 392, 440, 440, 392, 392, 350, 350, 330, 330, 394, 394, 262, 262, 392, 392, 350, 350, 330, 330, 294, 294, 392, 392, 350, 350, 330, 330],
[392, 392, 350, 350, 330, 330, 294, 294, 262, 262, 392, 392, 440, 440, 392, 392, 350, 350, 330, 330, 294, 294, 262, 262, 233, 207, 350, 350, 330, 330],
[175, 185, 196, 207, 220, 233, 247, 262, 277, 294, 311, 330, 350, 370, 392, 415, 440, 466, 493, 523, 554, 587, 622, 660, 699, 740, 784, 830, 880, 932]
]
which_set = int(input('Please input which song set you want(0~2)：'))

serdev = '/dev/ttyACM0'
formatter = lambda x: "%d" % x

s = serial.Serial(serdev)
s.write(bytes('z', 'UTF-8'))

for data in songset[which_set]:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("data transmit complete")
