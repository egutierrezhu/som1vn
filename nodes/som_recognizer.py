#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

"""
ann_recognizer.py is a MLP voice command recognizer
  publications:
    chatter (std_msgs/String) - text output
"""

import roslib; roslib.load_manifest('som1vn')
import rospy
from std_msgs.msg import String

from record1 import record_to_file
from features import mfcc
from somtester1 import *
import scipy.io.wavfile as wav

def som_recognizer():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('som_recognizer', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    words = ['backward','forward','go','left','right','stop']
           
    Nx = 30     #lenght of grid X
    Ny = Nx     #lenght of grid Y  
    print "SOM: " + str(Nx) + "x" + str(Ny)

    filename_som1 = rospy.get_param("~som1")
    filename_lab1 = rospy.get_param("~lab1")
    filename_test = rospy.get_param("~test")

    while not rospy.is_shutdown():
        rospy.loginfo("Record and Test")

        # Record to file
        record_to_file(filename_test)

        # Feed into SOM      
        testNet = testInit(Nx,Ny,filename_som1,filename_lab1)
        inputArray = extractFeature(filename_test)            
        outStr = feedToNetwork(words,inputArray,testNet)

        # Publish output string
        rospy.loginfo(str(outStr))
        pub.publish(str(outStr))
        rate.sleep()

if __name__ == '__main__':
    try:
        som_recognizer()
    except rospy.ROSInterruptException:
        pass
