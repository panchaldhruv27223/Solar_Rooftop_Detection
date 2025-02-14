## importing required library 
import cv2
import numpy 


def bounding_box(mask_image):
    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
    
    ## to see the mask image after thresholding
    # cv2.imshow("mask_images_after_threshold",binary_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(f"total number of rooftops in the image is : {len(contours)}")

    # List to store bounding boxes
    bounding_boxes = []

    # Loop through contours and get bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x+w, y+h))

    return bounding_boxes

def plot_bounding_box(image, bounding_box_cordinates):
    demo = image.copy()

    for bb in bounding_box_cordinates:
        x,y,w,h = bb
        cv2.rectangle(demo, (x,y),(x+w,y+h), (255,0,0), 2)

    cv2.imshow("draw_bounding_box", demo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return demo


if __name__ == "__main__":
    mask_image_path = "C:\\Users\\dhruv\\Documents\\DA-IICT\\Arpit_rana\\SAM_Model\\Codes\\Notebook\\images_1024\\masks\\0.jpg"
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    bb_cordinates = bounding_box(mask_image)
    # print(bb_cordinates)

    ### verify by ploting bounding box on the original image.
    image_ = cv2.imread("C:\\Users\\dhruv\\Documents\\DA-IICT\\Arpit_rana\\SAM_Model\\Codes\\Notebook\\images_1024\\images\\0.jpg")
    bb_image = plot_bounding_box(image_, bb_cordinates)
    cv2.imwrite("demo_bounding_box.jpg",bb_image)