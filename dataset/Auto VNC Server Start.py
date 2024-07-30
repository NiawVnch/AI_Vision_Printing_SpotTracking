sudo apt-get update
sudo apt-get install xfonts-base
sudo apt-get install tightvncserver
sudo apt install lxde -y



nano /home/linaro/start_vnc.sh

#####
#!/bin/bash
vncserver -localhost no
vncserver -kill :1
vncserver :1 
#####

chmod +x /home/linaro/start_vnc.sh


sudo nano /etc/systemd/system/vncserver.service

#####
[Unit]
Description=Start TightVNC server at startup
After=syslog.target network.target

[Service]
Type=forking
User=linaro
WorkingDirectory=/home/linaro
Environment=HOME=/home/linaro
ExecStart=/home/linaro/start_vnc.sh
ExecStop=/usr/bin/vncserver -kill :1

[Install]
WantedBy=multi-user.target
#####

sudo systemctl daemon-reload
sudo systemctl enable vncserver.service
sudo systemctl start vncserver.service

sudo systemctl status vncserver.service