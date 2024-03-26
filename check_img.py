import cv2
import glob


imgs = glob.glob('./dataset/*/images/*.png')
print(len(imgs))
for x in imgs:
    im = cv2.imread(x)
    cv2.imwrite(x, im)
