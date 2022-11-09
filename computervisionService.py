#!/usr/bin/env python
# coding: utf-8

# In[1]:


# create service file 
# in terminal type in:
# ~/Documents$ sudo vim /lib/systemd/system/MY_SERVICE.service

"""
My Service File
[Unit]
Description = ComputerVision
After= multi-user.target

[Service]
Type =simple
ExecStart=/home/robomaster/Documents/test.py
User = robomaster
WorkingDirectory= /home/robomaster/Documents
Restart = on-failure

[Install]
WantedBy= multi-user.target
~
~
~
~
~

"""

# enabling the service commands for terminal:
# sudo systemctl daemon-reload
# sudo systemctl enable MY_SERVICE.service
# sudo systemctl start MY_SERVICE.service

#starting the systemD service
# sudo systemctl start MY_SERVICE.service


# In[ ]:




