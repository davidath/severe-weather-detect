import sys
import os

with open(sys.argv[1]) as open_file:
   lines = open_file.readlines()
   os.system('clear')
   print '----------  PX ----------'
   for i in range(2, 11):
       print lines[i].replace('\n', '')
   raw_input()
   os.system('clear')
   print '----------  COND ----------'
   for i in range(13, 22):
       print lines[i].replace('\n', '')
   raw_input()
   os.system('clear')
