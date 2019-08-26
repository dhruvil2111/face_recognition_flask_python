import base64

from cv2 import cv2
from flask import Flask, render_template, Response
from camera import camera_stream

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frame():
    """Video streaming generator function."""
    while True:
        frame = camera_stream()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concate frame one by one and show result


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_frame')
def save_frame():
    frame = camera_stream()
    with open('dataset/hello.jpg', 'wb') as f:
        f.write(frame)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response('hello')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True, port=8080)
