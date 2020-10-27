import sys
import os

with open(sys.argv[1]) as open_file:
   lines = open_file.readlines()
   os.system('clear')
   print '----------  PX FULL ----------'
   for i in range(3, 12):
       print lines[i].replace('\n', '')
   raw_input()
   os.system('clear')
   print '----------  PX TEST ----------'
   for i in range(12, 21):
       print lines[i].replace('\n', '')
   raw_input()
   os.system('clear')
   print '----------  COND FULL ----------'
   for i in range(23, 32):
       print lines[i].replace('\n', '')
   raw_input()
   os.system('clear')
   print '----------  COND TEST ----------'
   for i in range(32, 41):
       print lines[i].replace('\n', '')
   raw_input()
   os.system('clear')
