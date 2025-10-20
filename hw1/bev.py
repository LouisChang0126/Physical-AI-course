import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.points = points if points else []

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###
        def rotation_matrix(theta, phi, gamma):
            theta, phi, gamma = np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(gamma)
            
            cx, sx = np.cos(theta), np.sin(theta)
            Rx = np.array([[1, 0,    0],
                           [0, cx, -sx],
                           [0, sx,  cx]])
            
            cy, sy = np.cos(phi),   np.sin(phi)
            Ry = np.array([[ cy, 0, sy],
                           [  0, 1,  0],
                           [-sy, 0, cy]])
            
            cz, sz = np.cos(gamma), np.sin(gamma)
            Rz = np.array([[cz, -sz, 0],
                           [sz,  cz, 0],
                           [ 0,   0, 1]])
            
            return Rz @ Ry @ Rx
        
        # 1. intrinsics
        cx = self.width / 2.0
        cy = self.height / 2.0
        f  = (self.width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
        
        # 2. extrinsics
        Center_front = np.array([0.0, 1.0, 0.0])
        Rotation_front = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])
        Center_BEV = np.array([0.0, 2.5, 0.0])
        Rotation_BEV = rotation_matrix(theta, phi, gamma) # BEV coordinates -> World coordinates

        # Transfer each point
        new_pixels = []
        
        for (u, v) in self.points:
            # world direction
            x_cam, y_cam = (u - cx) / f, -(v - cy) / f
            d_world = (Rotation_BEV @ [x_cam, y_cam, 1.0])

            # 3. BEV coordinates -> World coordinates
            if abs(d_world[1]) < 1e-6: # Remove ray parallel to ground(y=0)
                continue
            t = (0.0 - Center_BEV[1]) / d_world[1]
            if t < 0:  # intersect behind cam
                continue
            P_world = Center_BEV + t * d_world

            # 4. World coordinates -> Front coordinates
            Pc = (Rotation_front.T @ (P_world - Center_front))
            if Pc[2] < 0: # intersect behind cam
                continue

            u_ = (f * (Pc[0] / Pc[2])) + cx
            v_ = cy - (f * (Pc[1] / Pc[2]))
            u_ = int(np.clip(u_, 0, self.width - 1))
            v_ = int(np.clip(v_, 0, self.height - 1))
            
            new_pixels.append([u_, v_])

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = 90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
