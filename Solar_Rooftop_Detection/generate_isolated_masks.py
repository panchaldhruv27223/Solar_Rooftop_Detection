import torch

def generate_isolated_mask(mask, box):
    """"
    input = "mask" and "coordinates of bounding box"
    output = "Isolated mask.
    """
    x1, y1, x2, y2 = box
    print(x1, y1, x2, y2)

    isloated_mask = torch.zeros_like(mask)
    
    # print(mask[y1: y2, x1:x2].shape)
    
    isloated_mask[y1: y2, x1:x2] = mask[y1: y2, x1:x2]
    
    # print(np.unique(isloated_mask.numpy()))
    
    # cv2.imshow("new_mask",isloated_mask.numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print(isloated_mask.shape)

    return isloated_mask


if __name__ == "__main__":
    print("Hello from Dhruv Panchal.")