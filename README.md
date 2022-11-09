# RobomastersCVMidterm
Srikar Puri

For question 1 I only care about detecting the blue armor plates on the screen. So only detected and singled out the blue plates and ignored the red ones. 

for the object detection I did use thresholding to detect the plates

for the systemD service I chose to use simple because this service is long-running, whereas oneshot runs at only one time. the benefit to oneshot is that you can have multiple commands to run whereas with simple you can only run 1 command. In this case we are only running one command so simple is better

