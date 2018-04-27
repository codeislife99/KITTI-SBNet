import cv2
import numpy as np

image_hsv = None
pixel = (20,60,80)

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_bev[y,x]
        print(x,y)


def main():
    global image_bev, pixel

    # image_src = cv2.imread('./preprocessing/0001-bird-eye-view/00001.png')  # pick.py my.png
    image_bev = cv2.imread('../avod/0.jpg')  # pick.py my.png
    image_src = cv2.imread('../avod/avod/tests/datasets/Kitti/object/training/image_2/000142.png')
    image_label = open('../avod/avod/tests/datasets/Kitti/object/training/label_2/000142.txt','r+')

    image_bev2 = cv2.imread('../avod/5.jpg')  # pick.py my.png
    image_src2 = cv2.imread('../avod/avod/tests/datasets/Kitti/object/training/image_2/000216.png')
    image_label2 = open('../avod/avod/tests/datasets/Kitti/object/training/label_2/000217.txt','r+')

    img_center = (400, 700)
    cyclist_pixel = (365, 648)
    x_pixel_m_ratio = (img_center[0]-cyclist_pixel[0])/5.50
    y_pixel_m_ratio = (img_center[1]-cyclist_pixel[1])/3.49


    with image_label as openfileobject:
        for line in openfileobject:
            tx = line.split()
            print("category: {}".format(tx[0]))
            print("x is {}".format(tx[-4]))
            print("y is {}".format(tx[-3]))
            print("z is {}".format(tx[-2]))

            x_meter = tx[-4] # -3.49
            y_meter = tx[-2] # 5.50

            dx_pixel = float(x_meter) * x_pixel_m_ratio * (1) # - 26 pixel
            dy_pixel = float(y_meter) * y_pixel_m_ratio * (- 1) # - 17 pixel

            obj_x_pixel = img_center[0] + dx_pixel
            obj_y_pixel = img_center[1] + dy_pixel - 0.27 * y_pixel_m_ratio
            obj_xy = (int(obj_x_pixel),int(obj_y_pixel))
            print("label is on {}".format(obj_xy))
            if tx[0] == "Car":
                cv2.circle(image_bev, obj_xy, 3, (0, 0, 255), -1)
            else:
                cv2.circle(image_bev, obj_xy, 3, (0, 255, 0), -1)
    cv2.circle(image_bev,(365,648),3,(255,0,0),-1)

    if image_bev is None:
        print ("the image read is None............")
        return

    cv2.namedWindow('bev')
    cv2.setMouseCallback('bev', pick_color)

    # now click into the hsv img , and look at values:
    image_bev = cv2.cvtColor(image_bev,cv2.COLOR_BGR2HSV)
    cv2.imshow("bev",image_bev)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()