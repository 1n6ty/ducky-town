import argparse
import sys, traceback
import cv2
from matplotlib import pyplot as plt
import gym
import numpy as np
import pyglet
import time
import sys
import math
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

class Robot:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", default=None)
        parser.add_argument("--map-name", default="udem1")
        parser.add_argument("--distortion", default=False, action="store_true")
        parser.add_argument("--camera_rand", default=False, action="store_true")
        parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
        parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
        parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
        parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
        parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
        parser.add_argument("--seed", default=1, type=int, help="seed")
        args = parser.parse_args()

        if args.env_name and args.env_name.find("Duckietown") != -1:
            self.env = DuckietownEnv(
                seed=args.seed,
                map_name=args.map_name,
                draw_curve=args.draw_curve,
                draw_bbox=args.draw_bbox,
                domain_rand=args.domain_rand,
                frame_skip=args.frame_skip,
                distortion=args.distortion,
                camera_rand=args.camera_rand,
                dynamics_rand=args.dynamics_rand,
            )
        else:
            self.env = gym.make(args.env_name)

        self.env.reset()
        self.env.render()

        self.key_handler = key.KeyStateHandler()
        self.env.unwrapped.window.push_handlers(self.key_handler)
    
    def _2oneLine(self, coords):
        x1, y1, x2, y2 = 0, 0, 0, 0
        for i in coords:
            x1 += i[0]
            x2 += i[2]
            y1 += i[1]
            y2 += i[3]
        return [x1 / len(coords), y1 / len(coords), x2 / len(coords), y2 / len(coords)]
        

    def _rounding(self, left, right, up, down):
        return self._2oneLine(left), self._2oneLine(right), self._2oneLine(up), self._2oneLine(down)
        

    def isCross(self, posRobot, obs): # return coords of each side
        obs = cv2.GaussianBlur(cv2.cvtColor(obs, cv2.COLOR_BGR2HSV), (5, 5), 0)
        mask = cv2.inRange(obs, np.array([113,134,84]), np.array([138,215,215]))
        edges = cv2.Canny(mask, 75, 150)
        cv2.imshow('win', edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, maxLineGap = 50)
        try:
            x1, y1, x2, y2 = lines[0][0]
            if (y1 <= 280 and y2 >= 280) or (y1 >= 280 and y2 <= 280): #Cross was detected
                cv2.line(obs, (x1, y1), (x2, y2), (0, 255, 0), 2)
                leftRight = [line[0] for line in lines if 100 < line[0][1] < 500 and 100 < line[0][3] < 500]
                upDown = []
                for i in lines:
                    print(np.array(i[0], dtype = "int32"))
                    if np.array(i[0], dtype = "int32") in leftRight:
                        upDown.append(i[0])
                print(leftRight, upDown)
                left = [i for i in leftRight if i[0] - 320 < 0]
                right = [i for i in leftRight if i not in left]
                up = [i for i in upDown if i[1] - 240 < 0]
                down = [i for i in upDown if i not in up]
                left, right, up, down = self._rounding(left, right, up, down)
                obs = cv2.line(obs, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 2)
                #obs = cv2.line(obs, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 2)
                obs = cv2.line(obs, (up[0], up[1]), (up[2], up[3]), (0, 255, 0), 2)
                obs = cv2.line(obs, (down[0], down[1]), (down[2], down[3]), (0, 255, 0), 2)
                        
        except:
            traceback.print_exc(file=sys.stdout)
        cv2.imshow("win2", obs)
        if cv2.waitKey(33) == ord('e'):
            cv2.destroyAllWindows()
            sys.exit()

    def _manual(self, dt):
        wheel_distance = 0.102
        min_rad = 0.08

        action = np.array([0.0, 0.0])

        if self.key_handler[key.UP]:
            action += np.array([0.44, 0.0])
        if self.key_handler[key.DOWN]:
            action -= np.array([0.44, 0])
        if self.key_handler[key.LEFT]:
            action += np.array([0, 1])
        if self.key_handler[key.RIGHT]:
            action -= np.array([0, 1])

        v1 = action[0]
        v2 = action[1]
        if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
            delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
            v1 += delta_v
            v2 -= delta_v

        action[0] = v1
        action[1] = v2

        obs, reward, done, info = self.env.step(action)
        #print("step_count = %s, reward=%.3f" % (self.env.unwrapped.step_count, reward))

        self.isCross(self.env.cur_pos, obs)

        if done:
            print("done!")
            self.env.reset()

        self.env.render()

    def _pd(self, dt):
        lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        obs, reward, done, info = self.env.step([0.17, 5 * distance_to_road_center + 4 * angle_from_straight_in_rads])
        self.isCross(self.env.cur_pos, obs)
        self.env.render()

    def startLoopPd(self, Q):
        pyglet.clock.schedule_interval(self._pd, Q / self.env.unwrapped.frame_rate)
        pyglet.app.run()

    def startLoopManual(self, Q):
        pyglet.clock.schedule_interval(self._manual, Q / self.env.unwrapped.frame_rate)
        pyglet.app.run()
