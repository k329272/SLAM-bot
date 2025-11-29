# web_ui.py
import base64
import io
import threading
import time
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# global references (set by main)
GLOBAL = {
    'grid': None,
    'slam': None,
    'planner': None,
    'robot_pose': (0,0,0),
    'latest_scan': None,
    'path': None
}

@app.route('/')
def index():
    return render_template('index.html')

# serve static files (JS/CSS)
@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory('static', filename)

def encode_image(img):
    # img is numpy uint8 grayscale (H x W)
    colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # draw robot pose and path
    pose = GLOBAL.get('robot_pose', (0,0,0))
    grid = GLOBAL.get('grid')
    if grid is not None:
        # convert pose world -> cell -> px
        ix, iy = grid.world_to_cell(pose[0], pose[1])
        # convert to image coords
        H, W = img.shape
        px = ix
        py = grid.height - 1 - iy  # because we flipped earlier
        cv2.circle(colored, (px, py), 3, (0,0,255), -1)
    # path
    path = GLOBAL.get('path')
    if path:
        pts = []
        for x,y in path:
            ix, iy = grid.world_to_cell(x,y)
            px = ix
            py = grid.height - 1 - iy
            pts.append((px, py))
        for i in range(len(pts)-1):
            cv2.line(colored, pts[i], pts[i+1], (0,255,0), 1)
    # encode to png
    _, buffer = cv2.imencode('.png', colored)
    b64 = base64.b64encode(buffer).decode('ascii')
    return b64

def background_broadcaster(interval=0.5):
    while True:
        if GLOBAL['grid'] is not None:
            img = GLOBAL['grid'].to_image()
            b64 = encode_image(img)
            # robot pose
            pose = GLOBAL['slam'].pose if GLOBAL['slam'] else (0,0,0)
            socketio.emit('map', {'img': b64, 'pose': pose}, broadcast=True)
        time.sleep(interval)

@socketio.on('connect')
def on_connect():
    emit('connected', {'msg': 'ok'})

@socketio.on('set_goal')
def set_goal(data):
    # data contains pixel or world coords. We'll expect world x,y in meters
    gx = float(data.get('x'))
    gy = float(data.get('y'))
    # planner: plan path and store to GLOBAL['path']
    planner = GLOBAL.get('planner')
    slam = GLOBAL.get('slam')
    if planner is None or slam is None or GLOBAL['grid'] is None:
        emit('plan_result', {'ok': False, 'reason': 'not ready'})
        return
    start = (slam.pose[0], slam.pose[1])
    path = planner.plan(start, (gx, gy))
    if path is None:
        emit('plan_result', {'ok': False, 'reason': 'no_path'})
    else:
        GLOBAL['path'] = path
        emit('plan_result', {'ok': True, 'path': path})

def start_ui_thread(grid, slam, planner, host='0.0.0.0', port=5000):
    GLOBAL['grid'] = grid
    GLOBAL['slam'] = slam
    GLOBAL['planner'] = planner
    t = threading.Thread(target=background_broadcaster, daemon=True)
    t.start()
    # run flask socketio in this thread (non-blocking from caller)
    def run_app():
        socketio.run(app, host=host, port=port)
    th = threading.Thread(target=run_app, daemon=True)
    th.start()
