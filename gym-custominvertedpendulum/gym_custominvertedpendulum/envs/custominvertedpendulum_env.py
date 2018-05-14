"""
Classic inverted pendulum system
The pole does NOt exert any force onto the cart in the env, because
the stepper motor will damp that force out.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CustomInvertedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 0.236
        self.masspole = 0.032
        self.massball = 0.01 # golfball weight
        self.total_mass = (self.masspole + self.masscart + self.massball)
        self.length = 0.362
        self.radiusball = 0.0427/2 # golfball radius
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2.75
        self.tau = 0.02  # seconds between state updates
        self.inertia_pole = (self.masspole * self.length ** 2) / 3 +  \
                    self.massball * (self.length + self.radiusball) ** 2

        # Angle at which normalization occurs
        self.theta_threshold_radians = 15 * 2 * math.pi / 360
        self.x_threshold = 1.42/2 - 0.065/2 # half the rail length minus half the width of the cart

        # Scoring angle limit set to 2 * theta_threshold_radians
        lim = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-lim, lim)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        xacc  = force / self.total_mass
        t_gravity = self.length/2 * sintheta * self.masspole * self.gravity # torque due to gravity
        t_cart = self.length/2 * costheta * xacc * self.masspole # torque due to cart
        thetaacc = (-t_cart + t_gravity) / self.inertia_pole
        x_dot = x_dot + self.tau * xacc
        x  = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = angle_normalize(theta + self.tau * theta_dot)

        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold
        done = bool(done)

        if not done:
            reward = 2*np.cos(theta/2)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # start pole at bottom (180 degrees or pi radians)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.3
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
