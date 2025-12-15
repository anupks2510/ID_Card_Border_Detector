from ultralytics import YOLO
import cv2
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="YOLO-based Document Detector")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

   
    img = cv2.imread(args.image)
    if img is None:
        print(" Image not found Check the path.")
        return

    H, W = img.shape[:2]

    
    model = YOLO("yolov8n.pt")  

    results = model(img)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print(" No document detected!")
        return

   
    boxes = results.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = areas.argmax()

    x1, y1, x2, y2 = boxes[best_idx].astype(int)

    
    pad = int(0.02 * min(H, W))  

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)

   
    cropped = img[y1:y2, x1:x2]

   
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        vis,
        "Document Detected",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    base = os.path.basename(args.image)
    detected_path = os.path.join(args.output, f"{base}_detected.jpg")
    cropped_path = os.path.join(args.output, f"{base}_cropped.jpg")

    cv2.imwrite(detected_path, vis)
    cv2.imwrite(cropped_path, cropped)

    print("SUCCESS")
    print("Saved:", detected_path)
    print("Saved:", cropped_path)

  
    cv2.imshow("Detected Document", vis)
    cv2.imshow("Cropped Document", cropped)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
