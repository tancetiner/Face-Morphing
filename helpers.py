import numpy as np
import matplotlib.pyplot
import cv2
import scipy
import mediapipe as mp


def getFaceLandmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    # INITIALIZING OBJECTS
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # DETECT THE FACE LANDMARKS
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=min_detection_confidence, 
        min_tracking_confidence=min_tracking_confidence
    )
    #inputImage = cv2.imread(file)

    # Flip the image horizontally and convert the color space from BGR to RGB
    #image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Detect the face landmarks
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
      return [
        [int(l.x * image.shape[1]), int(l.y * image.shape[0])]
        for l in results.multi_face_landmarks[0].landmark
    ]
    else: return None


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, alpha):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    img2Rect = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] = (
        img[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] * (1 - mask) + imgRect * mask
    )
