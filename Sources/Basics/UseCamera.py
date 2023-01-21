import cv2


def print_text(image, text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    font_scale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font,
                        font_scale, color, thickness, cv2.LINE_AA)

    return image


def call_camera():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # print text on screen
        gray = print_text(gray, "Press 'q' to quit!")

        # Display the resulting frame
        cv2.imshow('Use Camera Demo', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
