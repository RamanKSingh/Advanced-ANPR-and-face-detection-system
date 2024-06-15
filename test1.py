import os
import cv2
import time

# Load the pre-trained Haar Cascade classifier for Indian license plates
plate_classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Create a new directory to store the license plate images
if not os.path.exists('vehicle plates'):
    os.makedirs('vehicle plates')

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Set the initial time
prev_time = time.time()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the frame
    plates = plate_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Process each detected license plate
    for (x, y, w, h) in plates:
        # Draw a bounding box around the license plate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Check if 3 seconds have passed since the last photo
        current_time = time.time()
        if current_time - prev_time >= 3:
            # Extract the license plate from the frame
            plate = gray[y:y+h, x:x+w]

            # Save the license plate image to a file
            filename = f'vehicle plates/plate_{len(os.listdir("vehicle plates"))+1}.jpg'
            cv2.imwrite(filename, plate)

            # Set the new previous time
            prev_time = current_time

    # Display the frame with bounding boxes around license plates
    cv2.imshow('Frame', frame)

    # Wait for a key event
    key = cv2.waitKey(1)

    # Check if the user has pressed the 'q' key or closed the window
    if key == ord('q') or cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close all windows4
cap.release()
cv2.destroyAllWindows()
