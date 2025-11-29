# Remember to chmod +x 755 before trying to run!
sudo apt update
sudo apt install python3-pip python3-opencv pigpio
pip3 install rplidar numpy scipy flask flask-socketio eventlet opencv-python
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
