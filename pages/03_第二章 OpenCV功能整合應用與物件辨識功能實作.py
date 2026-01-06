import streamlit as st

st.title("第二章 OpenCV功能整合應用與物件辨識功能實作")

# 在這裡添加實驗一的具體內容，如圖表、數據等

st.image("image/19.png")
st.markdown("""
<h2>YOLO辨識網站</h2>

<ol style="font-size:18px;">
<li><a href="https://app.roboflow.com/imagerecognitiontest-uxpxd/my-first-project-gy9er/models/my-first-project-gy9er/2" target="_blank">辨識網站</a></li>
</ol>
""", unsafe_allow_html=True)

st.header("2.1 RL建模與訓練")
sample_video = open("image/1.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)

st.header("2.2 結果展示")
sample_video = open("image/28.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)

st.subheader("""main.py Code""")
code = '''"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
#from final.env import ArmEnv
from final.env_3link import ArmEnv
from final.rl import DDPG
import pyglet


MAX_EPISODES = 1200
MAX_EP_STEPS = 400


ON_TRAIN = True #True or False

env = ArmEnv()
rl = DDPG(env.action_dim, env.state_dim, env.action_bound)


def train():
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0
        for _ in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            if rl.memory_full:
                rl.learn()
            s = s_
            ep_r += r
            if done:
                break
        print(f"Ep {ep:03d} | R = {ep_r:.2f}")
    rl.save()


def eval():
    rl.restore()
    env.render()
    s = env.reset()

    def update(dt):
        nonlocal s
        a = rl.choose_action(s)
        s, _, _ = env.step(a)

    pyglet.clock.schedule_interval(update, 1/60.)
    pyglet.app.run()


if ON_TRAIN:
    train()
else:
    eval()
'''
st.code(code, language='python')

st.subheader("""env_3link.py Code""")
code = '''import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None
    dt = 0.1
    action_bound = [-1, 1]

    goal = {'x': 200., 'y': 200., 'l': 40}

    state_dim = 9
    action_dim = 3

    def __init__(self):
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)]
        )
        self.arm_info['l'] = 80
        self.arm_info['r'] = np.pi / 6

        self.on_goal = 0
        self.prev_action = np.zeros(self.action_dim)

    def step(self, action):
        action = np.clip(action, *self.action_bound)

        # ===== anti-jitter =====
        alpha = 0.2
        action = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = action

        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2

        # ===== forward kinematics =====
        p0 = np.array([200., 200.])
        r1, r2, r3 = self.arm_info['r']
        l1, l2, l3 = self.arm_info['l']

        p1 = p0 + np.array([np.cos(r1), np.sin(r1)]) * l1
        p2 = p1 + np.array([np.cos(r1+r2), np.sin(r1+r2)]) * l2
        finger = p2 + np.array([np.cos(r1+r2+r3), np.sin(r1+r2+r3)]) * l3

        # ===== distances (same as 2-link format) =====
        dist1 = [(self.goal['x'] - p2[0]) / 400,
                 (self.goal['y'] - p2[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] - finger[1]) / 400]

        # ===== reward =====
        r = -np.sqrt(dist2[0]**2 + dist2[1]**2)
        r -= 0.05 * np.linalg.norm(action)

        done = False
        if abs(dist2[0]) < self.goal['l']/400 and abs(dist2[1]) < self.goal['l']/400:
            r += 1.
            self.on_goal += 1
            if self.on_goal > 50:
                done = True
        else:
            self.on_goal = 0

        # ===== 9D state =====
        s = np.concatenate((
            p2 / 200,
            finger / 200,
            dist1 + dist2,
            [1. if self.on_goal else 0.]
        ))

        assert s.shape[0] == self.state_dim
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand() * 400
        self.goal['y'] = np.random.rand() * 400
        self.arm_info['r'] = 2 * np.pi * np.random.rand(3)

        self.prev_action = np.zeros(self.action_dim)
        self.on_goal = 0

        p0 = np.array([200., 200.])
        r1, r2, r3 = self.arm_info['r']
        l1, l2, l3 = self.arm_info['l']

        p1 = p0 + np.array([np.cos(r1), np.sin(r1)]) * l1
        p2 = p1 + np.array([np.cos(r1+r2), np.sin(r1+r2)]) * l2
        finger = p2 + np.array([np.cos(r1+r2+r3), np.sin(r1+r2+r3)]) * l3

        dist1 = [(self.goal['x'] - p2[0]) / 400,
                 (self.goal['y'] - p2[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] - finger[1]) / 400]

        s = np.concatenate((
            p2 / 200,
            finger / 200,
            dist1 + dist2,
            [0.]
        ))

        assert s.shape[0] == self.state_dim
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)


# ========================= Viewer =========================

class Viewer(pyglet.window.Window):
    def __init__(self, arm_info, goal):
        super().__init__(400, 400, '3-Link Arm', resizable=False, vsync=True)

        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.arm_info = arm_info
        self.goal = goal
        self.center = np.array([200, 200])
        self.bar_thickness = 6

        self.batch = pyglet.graphics.Batch()

        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2f', [0]*8), ('c3B', (255, 0, 0)*4))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2f', [0]*8), ('c3B', (255, 0, 0)*4))
        self.arm3 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2f', [0]*8), ('c3B', (255, 0, 0)*4))

        self.goal_box = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                       ('v2f', [0]*8), ('c3B', (80, 120, 255)*4))

    def thick_line(self, p0, p1, t):
        v = p1 - p0
        v /= np.linalg.norm(v) + 1e-6
        n = np.array([-v[1], v[0]]) * t
        return (*(p0+n), *(p0-n), *(p1-n), *(p1+n))

    def on_draw(self):
        self.clear()
        self._update()
        self.batch.draw()

    def _update(self):
        r1, r2, r3 = self.arm_info['r']
        l1, l2, l3 = self.arm_info['l']

        p0 = self.center
        p1 = p0 + np.array([np.cos(r1), np.sin(r1)]) * l1
        p2 = p1 + np.array([np.cos(r1+r2), np.sin(r1+r2)]) * l2
        p3 = p2 + np.array([np.cos(r1+r2+r3), np.sin(r1+r2+r3)]) * l3

        t = self.bar_thickness
        self.arm1.vertices = self.thick_line(p0, p1, t)
        self.arm2.vertices = self.thick_line(p1, p2, t)
        self.arm3.vertices = self.thick_line(p2, p3, t)

        gx, gy, gl = self.goal['x'], self.goal['y'], self.goal['l']/2
        self.goal_box.vertices = (
            gx-gl, gy-gl,
            gx-gl, gy+gl,
            gx+gl, gy+gl,
            gx+gl, gy-gl
        )

    def on_mouse_motion(self, x, y, dx, dy):
        self.goal['x'] = x
        self.goal['y'] = y
'''
st.code(code, language='python')

st.subheader("""rl.py Code""")
code = '''import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')

'''
st.code(code, language='python')

st.image("image/3.png")
st.image("image/4.png")
st.image("image/5.png")
st.image("image/6.png")
st.image("image/7.png")
st.image("image/8.png")
st.image("image/9.png")
st.image("image/10.png")
st.image("image/11.png")
st.image("image/12.png")
st.image("image/13.png")
st.image("image/14.png")
st.image("image/15.png")
st.image("image/16.png")
st.image("image/17.png")
st.image("image/18.png")
st.image("image/19.jpg")
st.image("image/20.jpg")
st.image("image/21.jpg")
st.image("image/22.jpg")
st.image("image/23.jpg")
st.image("image/24.jpg")
st.image("image/25.jpg")
st.image("image/26.jpg")
st.image("image/27.jpg")
st.image("image/28.jpg")
st.image("image/29.jpg")