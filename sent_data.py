# the program is just use to send the song information
import serial
import time
waitTime = 0.1

songset = ['_C4', '_C4', '_G4', '_G4', '_A4', '_A4', '_G4', '_G4', '_F4', '_F4', '_E4', '_E4', '_D4', '_D4', '_C4', '_C4', '_G4', '_G4', '_F4', '_F4', '_E4', '_E4', '_D4', '_D4', '_G4', '_G4', '_F4', '_F4', '_E4', '_E4']
which_set = int(input('Please input which song set you want(0~2)：'))

serdev = '/dev/ttyACM1'

s = serial.Serial(serdev)

while 1:
  line = s.readline()
  if line.decode('utf-8') == "start":
    break

for data in songset:
  s.write(bytes((data), 'UTF-8'))
  #print(bytes(data, 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("data transmit complete")
