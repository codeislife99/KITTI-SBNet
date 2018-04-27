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

    # image_bev = cv2.imread('../avod/0.jpg')  # pick.py my.png
    # image_label = open('../avod/avod/tests/datasets/Kitti/object/training/label_2/000142.txt','r+')

    # image_bev = cv2.imread('../avod/5.jpg')  # pick.py my.png
    # image_label = open('../avod/avod/tests/datasets/Kitti/object/training/label_2/000217.txt','r+')

    # image_bev = cv2.imread('../../bev_samples/000003_1.jpg')  # pick.py my.png
    # image_label = open('../avod/000003.txt','r+')

    image_bev = cv2.imread('../../bev_samples/000021_1.jpg')  # pick.py my.png
    image_label = open('../avod/000021.txt', 'r+')
    cv2.imshow("raw pic",image_bev)
    img_center = (400.1475, 704.2929)
    cyclist_pixel = (365.0, 648.0)
    cyclist_location = (5.50,3.49)

    ped_pixel = (425,347)
    ped_location = (2.54,35.54)
    x_pixel_m_ratio = (img_center[0]-ped_pixel[0])/ped_location[0]
    y_pixel_m_ratio = (img_center[1]-ped_pixel[1])/ped_location[1]

    with image_label as openfileobject:
        for line in openfileobject:
            tx = line.split()
            print("category: {}".format(tx[0]))
            print("x is {}".format(tx[-4]))
            print("y is {}".format(tx[-3]))
            print("z is {}".format(tx[-2]))

            print("height is {}".format(tx[-7]))
            print("width is {}".format(tx[-6]))
            print("length is {}".format(tx[-5]))

            # object center meters location to pixel location
            x_meter = tx[-4]
            y_meter = tx[-2]
            dx_pixel = float(x_meter) * x_pixel_m_ratio * (- 1)
            dy_pixel = float(y_meter) * y_pixel_m_ratio * (- 1)

            # object center point
            obj_x_pixel = img_center[0] + dx_pixel
            # obj_y_pixel = img_center[1] + dy_pixel - 0.27 * y_pixel_m_ratio
            obj_y_pixel = img_center[1] + dy_pixel - 0.27
            obj_xy = (int(obj_x_pixel), int(obj_y_pixel))

            # dimension meters location to pixel location
            x_pixel_dim = abs(float(tx[-6]) * x_pixel_m_ratio)
            y_pixel_dim = abs(float(tx[-5]) * y_pixel_m_ratio)
            half_x_pixel_dim = x_pixel_dim/2.0
            half_y_pixel_dim = y_pixel_dim/2.0


            # bbox points
            obj_left_down_pixel_bbox_point = (int(obj_x_pixel - half_x_pixel_dim), int(obj_y_pixel + half_y_pixel_dim))
            obj_right_up_pixel_bbox_point = (int(obj_x_pixel + half_x_pixel_dim), int(obj_y_pixel - half_y_pixel_dim))

            # draw bbox (now without rotation)
            cv2.rectangle(image_bev, obj_left_down_pixel_bbox_point, obj_right_up_pixel_bbox_point,(255,255,0),3)
            print("label is on {}".format(obj_xy))
            cv2.circle(image_bev, obj_xy, 3, (0, 0, 255), -1)


    # cv2.imwrite("image0_1",image_bev)
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