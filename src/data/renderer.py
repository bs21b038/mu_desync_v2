import torch
import math


class TensorRenderer:

    def __init__(
        self,
        image_size=128,
        xlim=(-0.8, 0.8),
        ylim=(-0.8, 0.8),
        arm_radius=4,
        target_size=5
    ):

        self.image_size = image_size

        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim

        self.arm_radius = arm_radius
        self.target_size = target_size

        # RGB colors
        self.BLUE = torch.tensor([0.0, 0.0, 1.0])
        self.GREEN = torch.tensor([0.0, 1.0, 0.0])
        self.RED = torch.tensor([1.0, 0.0, 0.0])
        self.BLACK = torch.tensor([0.0, 0.0, 0.0])

    # ---------------------------------------------------
    # coordinate → pixel
    # ---------------------------------------------------
    def to_pixel(self, x, y):

        px = int(
            (x - self.xmin)
            / (self.xmax - self.xmin)
            * (self.image_size - 1)
        )

        py = int(
            (self.ymax - y)
            / (self.ymax - self.ymin)
            * (self.image_size - 1)
        )

        # clip for safety
        px = max(0, min(self.image_size - 1, px))
        py = max(0, min(self.image_size - 1, py))

        return px, py

    # ---------------------------------------------------
    # draw filled circle with black edge
    # ---------------------------------------------------
    def draw_circle(
        self,
        img,
        px,
        py,
        radius,
        color,
        edge_color=None
    ):

        H = self.image_size
        W = self.image_size

        for y in range(py - radius, py + radius + 1):

            for x in range(px - radius, px + radius + 1):

                if x < 0 or x >= W or y < 0 or y >= H:
                    continue

                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)

                # outer edge
                if edge_color is not None:
                    if radius - 1 <= dist <= radius:
                        img[:, y, x] = edge_color

                # inner fill
                if dist < radius - 1:
                    img[:, y, x] = color

    # ---------------------------------------------------
    # draw upward triangle with black edge
    # ---------------------------------------------------
    def draw_triangle(
        self,
        img,
        px,
        py,
        size,
        color,
        edge_color=None
    ):

        H = self.image_size
        W = self.image_size

        for dy in range(size):

            row_y = py - size//2 + dy

            if row_y < 0 or row_y >= H:
                continue

            half_width = dy

            for dx in range(-half_width, half_width + 1):

                x = px + dx

                if x < 0 or x >= W:
                    continue

                # edge
                if edge_color is not None:

                    if (
                        dx == -half_width
                        or dx == half_width
                        or dy == size - 1
                    ):
                        img[:, row_y, x] = edge_color

                    else:
                        img[:, row_y, x] = color

                else:
                    img[:, row_y, x] = color

    # ---------------------------------------------------
    # render full frame
    # ---------------------------------------------------
    def render(self, coords, target):

        # white background
        img = torch.ones(
            3,
            self.image_size,
            self.image_size
        )

        xL, yL, xR, yR = coords
        xT, yT = target

        # convert to pixels
        pxL, pyL = self.to_pixel(xL, yL)
        pxR, pyR = self.to_pixel(xR, yR)
        pxT, pyT = self.to_pixel(xT, yT)

        # draw left arm
        self.draw_circle(
            img,
            pxL,
            pyL,
            self.arm_radius,
            self.BLUE,
            self.BLACK
        )

        # draw right arm
        self.draw_circle(
            img,
            pxR,
            pyR,
            self.arm_radius,
            self.GREEN,
            self.BLACK
        )

        # draw target
        self.draw_triangle(
            img,
            pxT,
            pyT,
            self.target_size,
            self.RED,
            self.BLACK
        )

        return img