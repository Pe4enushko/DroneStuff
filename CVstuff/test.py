import cv2
import numpy as np


# =========================
# НАСТРОЙКИ
# =========================

SOURCE_MODE = "camera"   # "image" или "camera"

IMAGE_PATH = "field.jpg"
CAMERA_INDEX = 0

USE_WHITE_MASK = True   # True — искать белую/светлую разметку
MIN_AREA = 1500


# =========================
# ФУНКЦИИ
# =========================

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.reshape(4, 2).astype("float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype="float32"
    )


def find_rectangle(frame: np.ndarray) -> np.ndarray | None:
    """
    Ищет самый крупный четырёхугольник на кадре.
    Возвращает 4 точки или None.
    """

    if USE_WHITE_MASK:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 80, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        work = mask

    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        work = edges

    contours, _ = cv2.findContours(
        work,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][1]


def draw_detected_rectangle(frame: np.ndarray) -> np.ndarray:
    output = frame.copy()

    quad = find_rectangle(frame)

    if quad is None:
        cv2.putText(
            output,
            "Rectangle not found",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        return output

    ordered = order_points(quad)

    # Рисуем найденную фигуру
    cv2.polylines(
        output,
        [ordered.astype(np.int32)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3
    )

    # Рисуем углы
    labels = ["TL", "TR", "BR", "BL"]

    for point, label in zip(ordered, labels):
        x, y = point.astype(int)

        cv2.circle(output, (x, y), 7, (0, 0, 255), -1)

        cv2.putText(
            output,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return output


# =========================
# ЗАПУСК
# =========================

def run_image():
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {IMAGE_PATH}")

    output = draw_detected_rectangle(frame)

    cv2.imshow("detected figure", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру: {CAMERA_INDEX}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output = draw_detected_rectangle(frame)

        cv2.imshow("detected figure", output)

        # ESC — выход
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if SOURCE_MODE == "image":
        run_image()
    elif SOURCE_MODE == "camera":
        run_camera()
    else:
        raise ValueError('SOURCE_MODE должен быть "image" или "camera"')