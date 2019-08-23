from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


#  for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'


def camera_stream():
    args = {'detection_method': 'hog',
            'display': 1,
            # 'input' : '/home/jugal/Desktop/Face Recognition/videos/Avengers.mp4',
            'encodings': 'encodings_international_celebrity.pickle',
            'save_to': 'dataset/',
            'output': None}

    # %%

    # load the known faces and embeddings
    print("Loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    # %%

    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("Starting video stream...")
    vs = VideoStream(src=0).start()
    writer = None
    time.sleep(2.0)

    # %%

    count = 0
    while True:
        frame = vs.read()

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=320)
        r = frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (255, 255, 255), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

        # if the video writer is None *AND* we are supposed to write the output video to disk initialize the writer
        if writer is None and args["output"] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 10,
                                     (640, 480), True)

        # if the writer is not None, write the frame with recognized faces to disk
        if writer is not None:
            writer.write(frame)

        # check to see if we are supposed to display the output frame to the screen
        if args["display"] > 0:
            # cv2.imshow("Frame", frame)
            for name in names:
                if name == 'Unknown':
                    cv2.imwrite(args["save_to"] + str(count) + ".jpg", frame)
                    count += 1
        # Display the resulting frame in browser
        return cv2.imencode('.jpg', frame)[1].tobytes()
