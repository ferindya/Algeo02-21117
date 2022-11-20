import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    h = image.shape[0]
    w = image.shape[1]
    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized