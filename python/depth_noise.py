#!/usr/bin/env python3
"""
Make the depth image of the simulated camera look like the one of a real RealSense: the simulated one is perfect, and
what works on it says little. Only for simulation: on the real arm the camera does this by itself.

  noise that grows with the square of the distance and is shared by neighbouring pixels, as stereo matching does
  a slow error across the image, that bends the plane of the table a little
  no depth where one of the two imagers does not see: the stereo shadow on the side of everything that stands out
  edges: pixels lost, or halfway between the two depths ("flying pixels")
  isolated pixels lost, and depth in steps of 1 mm

Usage:
  python3 depth_noise.py [--strength 1.0]
  python3 pick_and_place.py --depth-topic /camera/aligned_depth_to_color/image_noisy
"""

import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from scipy import ndimage
from sensor_msgs.msg import CameraInfo, Image

DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
NOISY_TOPIC = '/camera/aligned_depth_to_color/image_noisy'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'

NOISE_PER_M2 = 0.006            # m of noise (1 sigma) per m^2 of distance: 1.2 mm at 0.45 m, D435 at about 0.8 mm at best
NOISE_BLUR = 1.0                # pixels: how far the noise is shared
SLOW_ERROR = 0.0005             # m (1 sigma) of the error across the image
SLOW_GRID = 8                   # cells of that error along the image
BASELINE = 0.05                 # m between the two imagers
EDGE_JUMP = 0.005               # m of depth between neighbouring pixels that makes an edge
EDGE_LOST = 0.5                 # of the pixels on an edge
LOST_PIXELS = 0.002             # of all the pixels
DEPTH_STEP = 0.001              # m


class DepthNoise(Node):

    def __init__(self, strength):
        super().__init__('depth_noise')
        self.strength = strength
        self.rng = np.random.default_rng()
        self.fx = None
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Image, NOISY_TOPIC, qos_profile_sensor_data)

    def on_camera_info(self, msg):
        self.fx = msg.k[0]

    def noisy(self, depth):
        valid = np.isfinite(depth) & (depth > 0)
        z = np.where(valid, depth, 0.0)
        field = ndimage.gaussian_filter(self.rng.standard_normal(z.shape), NOISE_BLUR) * 2 * np.sqrt(np.pi) * NOISE_BLUR
        slow = ndimage.zoom(self.rng.standard_normal((SLOW_GRID, SLOW_GRID * z.shape[1] // z.shape[0])),
                            (z.shape[0] / SLOW_GRID, z.shape[1] / (SLOW_GRID * z.shape[1] // z.shape[0])), order=3)
        out = z + self.strength * (NOISE_PER_M2 * z ** 2 * field + SLOW_ERROR * slow[:z.shape[0], :z.shape[1]])

        # a pixel the left imager sees is hidden from the right one by anything nearer on its right
        with np.errstate(divide='ignore'):
            disparity = np.where(valid, self.fx * BASELINE / np.maximum(z, 1e-6), 0.0)
        shifted = np.arange(z.shape[1]) - disparity
        next_on_right = np.minimum.accumulate(shifted[:, ::-1], axis=1)[:, ::-1]
        hidden = np.zeros(z.shape, bool)
        hidden[:, :-1] = shifted[:, :-1] > next_on_right[:, 1:] + 0.5

        jump = np.maximum(np.abs(np.diff(z, axis=0, prepend=z[:1])), np.abs(np.diff(z, axis=1, prepend=z[:, :1])))
        edge = jump > EDGE_JUMP
        near = ndimage.minimum_filter(z, size=3)
        far = ndimage.maximum_filter(z, size=3)
        out = np.where(edge, (near + far) / 2 + (far - near) / 2 * self.rng.uniform(-1, 1, z.shape), out)

        lost = hidden | (edge & (self.rng.random(z.shape) < EDGE_LOST * min(self.strength, 1 / EDGE_LOST)))
        lost |= self.rng.random(z.shape) < LOST_PIXELS * self.strength
        out = np.round(out / DEPTH_STEP) * DEPTH_STEP
        out[lost | ~valid] = 0.0
        return out.astype(np.float32)

    def on_depth(self, msg):
        if self.fx is None or msg.encoding != '32FC1':
            return
        depth = np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding=msg.encoding,
                    is_bigendian=msg.is_bigendian, step=msg.step, data=self.noisy(depth).tobytes())
        self.publisher.publish(out)


def main():
    parser = argparse.ArgumentParser(description='Make the simulated depth image look like the one of a real camera')
    parser.add_argument('--strength', type=float, default=1.0, help='scale of the noise: 0 leaves only the shadows and the edges')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init(args=sys.argv)
    node = DepthNoise(args.strength)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
