import pickle
import cv2
import numpy as np

with open("data/label_mapping.pkl", "rb") as f:
    label_map = pickle.load(f)

print(label_map)

cap = cv2.VideoCapture("data/train/Băn khoăn/492077.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_std = np.std(frame)
    print(f"Current STD: {current_std}")

    if current_std > 90 or current_std < 10:
        continue

    cv2.imshow("Video Playback", frame)
    if cv2.waitKey(30) & 0xff == ord("q"):
        break

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
cap.release()
cv2.destroyAllWindows()