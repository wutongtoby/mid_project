# the program is just use to send the song information
import serial
import time
waitTime = 0.1

songset1 = ['_C4', 'CS4', '_D4', 'DS4', '_E4', '_F4', 'FS4', '_G4', 'GS4', '_A4', 'AS4', '_B4', '_C5', 'CS5', '_D5', 'DS5', '_E5', '_F5', 'FS5', '_G5', 'GS5', '_E5', '_C5', '_E5', '_F5', '_D5', '_B4', '_G4', '_C5', '_C5']
songset2 = ['_A4', '_D5', '_E5', 'FS5', '_E5', '_D5', '_D5', '_A4', '_D5', '_E5', 'FS5', '_E5', '_A5', 'FS5', 'XXX', 'FS5', '_G5', '_A5', 'XXX', '_F4', '_F4', '_G4', '_A4', 'AS4', '_A4', '_G4', '_F4', '_E4', '_F4', '_C5']
which_set = int(input('Please input which song set you want(0~1)：'))

songset = [songset1, songset2]

serdev = '/dev/ttyACM0'

s = serial.Serial(serdev)
s.write(bytes(('z'), 'UTF-8'))

for data in songset[int(which_set)]:
  s.write(bytes((data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("data transmit complete")
