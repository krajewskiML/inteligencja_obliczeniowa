"""In this module we define the custom environment for the project. It will simulate a body which will be controlled by
the agent. The environment will be a 2D space with a 2D body. The body will have a position and a velocity. The agent
will be able to apply a force to the body. The goal of the agent is to move the body in such a way that it avoids
arrows that will be shot in its direction"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional
import pygame

import Box2D
from Box2D import b2CircleShape, b2ContactListener, b2FixtureDef, b2PolygonShape, b2Vec2


FPS = 50
SCALE = 50.0  # affects how fast-paced the game is, forces should be adjusted as well

ENGINE_POWER = 200
CENTER_REWARD = 50
ARROW_REWARD = 10
ALIVE_REWARD = 5

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder
INITIAL_SHOT_FREQ = 50
INCREASE_SHOT_FREQ_AFTER = 5
HIT_REWARD = -1000

AVOIDER_RADIUS = 10
ARROW_RADIUS = 12
ARROW_SPEED = 200

VIEWPORT_W = 1200
VIEWPORT_H = 800

class ContactDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if not (
            self.env.avoider == contact.fixtureA.body
            or self.env.avoider == contact.fixtureB.body
        ):
            return
        else:
            self.env.reward = HIT_REWARD
            self.env.game_over = True

    def EndContact(self, contact):
        pass


class ArrowAvoider(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        gravity: float = 0.0,
        cones: int = 8
    ):
        assert cones > 0, "Number of cones must be greater than 0"

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.avoider: Optional[Box2D.b2Body] = None
        self.particles = []
        self.cones = cones

        self.prev_reward = None

        low = np.array(
            [
                # position bounds
                -1.0,
                -1.0,
                # velocity bounds are 5x rated speed
                -5.0,
                -5.0,
            ] +
            [
                0.0 for _ in range(cones)
            ] +
            [
                -1.0 for _ in range(cones)
            ]
        ).astype(np.float32)

        high = np.array(
            [
                # position bounds
                1.0,
                1.0,
                # velocity bounds are 5x rated speed
                5.0,
                5.0,
            ] +
            [
                1.0 for _ in range(cones)
            ] +
            [
                1.0 for _ in range(cones)
            ]
        ).astype(np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # 5 actions: do nothing, apply force to the left, apply force to the right, apply force up, apply force down
        # I also consider the case where the agent can not apply force down and then use gravity to make it fall
        self.action_space = spaces.Discrete(5)

        self.render_mode = render_mode

    def _destroy(self):
        if self.avoider is None:
            return
        self.world.contactListener = None
        # self._clean_particles(True)
        self.world.DestroyBody(self.avoider)
        self.avoider = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.shooting_freq = INITIAL_SHOT_FREQ
        self.last_shot_time = 0
        self.shot_counter = 0
        self.game_over = False
        self.prev_shaping = None
        self.steps = 0
        self.arrows = []

        # walls of the screen (left, right, top, bottom) covering whole edges of the screen
        self.walls = [
            # left wall starting at (0, 0) with width 1 and height VIEWPORT_H
            self.world.CreateStaticBody(
                position=(-1, -1),
                angle=0.0,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(1, VIEWPORT_H)),
                    friction=0.0,
                    restitution=0.0,
                ),
            ),
            # right wall starting at (VIEWPORT_W, 0) with width 1 and height VIEWPORT_H
            self.world.CreateStaticBody(
                position=(VIEWPORT_W + 1, 0),
                angle=0.0,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(1, VIEWPORT_H)),
                    friction=0.0,
                    restitution=0.0,
                ),
            ),
            # top wall starting at (0, 0) with width VIEWPORT_W and height 1
            self.world.CreateStaticBody(
                position=(-1, -1),
                angle=0.0,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(VIEWPORT_W, 1)),
                    friction=0.0,
                    restitution=0.0,
                ),
            ),
            # bottom wall starting at (0, VIEWPORT_H) with width VIEWPORT_W and height 1
            self.world.CreateStaticBody(
                position=(0, VIEWPORT_H + 1),
                angle=0.0,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(VIEWPORT_W, 1)),
                    friction=0.0,
                    restitution=0.0,
                ),
            ),
        ]

        # avoider, creating body that will avoid the arrows
        initial_x = np.random.uniform(0, VIEWPORT_W)
        initial_y = np.random.uniform(0, VIEWPORT_H)
        self.avoider = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=AVOIDER_RADIUS / SCALE, pos=(0, 0)),
                density=5.0,
                friction=0.0,
                restitution=0.0,
            ),
        )

        # apply random force to avoider
        self.avoider.ApplyForceToCenter(
            (
                np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )
        return self.get_observation(), {}

    def shoot_arrow(self):
        if self.steps - self.last_shot_time == self.shooting_freq:
            self.last_shot_time = self.steps
            self.shot_counter += 1
            # create arrow on random wall of the screen
            # apply force to the arrow that is pointing towards the avoider
            # add arrow to the list of arrows

            # get random wall
            wall = np.random.randint(0, 4)
            # get random position on the wall
            if wall == 0:
                # top wall
                x = np.random.uniform(0, VIEWPORT_W)
                y = 0
            elif wall == 1:
                # bottom wall
                x = np.random.uniform(0, VIEWPORT_W)
                y = VIEWPORT_H
            elif wall == 2:
                # left wall
                x = 0
                y = np.random.uniform(0, VIEWPORT_H)
            else:
                # right wall
                x = VIEWPORT_W
                y = np.random.uniform(0, VIEWPORT_H)
            # create arrow
            arrow = self.world.CreateKinematicBody(
                position=(x, y),
                angle=0.0,
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=ARROW_RADIUS / SCALE, pos=(0, 0)),
                    density=0.1,
                    friction=0.0,
                    restitution=0.0,
                ),
            )
            # get vector pointing towards the avoider
            avoider_pos = self.avoider.position
            arrow_pos = arrow.position
            direction = (avoider_pos.x - arrow_pos.x, avoider_pos.y - arrow_pos.y)
            # normalize vector
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction = direction / norm
            # set arrow velocity
            arrow.linearVelocity = ARROW_SPEED * b2Vec2(direction[0], direction[1])
            # add arrow to the list of arrows
            self.arrows.append(arrow)

    def update_arrows(self):
        # remove arrows that are out of the screen
        for arrow in self.arrows:
            arrow_pos = arrow.position
            if (
                arrow_pos.x < 0
                or arrow_pos.x > VIEWPORT_W
                or arrow_pos.y < 0
                or arrow_pos.y > VIEWPORT_H
            ):
                self.arrows.remove(arrow)
                self.world.DestroyBody(arrow)
            else:
                # check if arrow hit the avoider
                distance = (arrow_pos - self.avoider.position).length
                if (distance - AVOIDER_RADIUS) < ARROW_RADIUS:
                    self.arrows.remove(arrow)
                    self.world.DestroyBody(arrow)
                    self.game_over = True
                    self.reward = HIT_REWARD
                    return

    def get_observation(self):
        # first let's get the position of the avoider
        avoider_pos = self.avoider.position
        # let's also get the velocity of the avoider
        avoider_vel = self.avoider.linearVelocity
        self_observation = np.array(
            [
                ((avoider_pos.x / VIEWPORT_W) - 0.5) * 2,
                ((avoider_pos.y / VIEWPORT_H) - 0.5) * 2,
                avoider_vel.x / SCALE,
                avoider_vel.y / SCALE,
            ]
        )

        # now create a numpy array with the shape of number of cones and fill it with ones
        arrow_distances = np.zeros(self.cones)
        arrow_directions = np.zeros(self.cones)
        # now iterate over arrows and get the distance to the closest arrow in each cone direction
        for arrow in self.arrows:
            arrow_pos = arrow.position
            # get the angle of the arrow relative to the avoider
            angle = np.arctan2(
                arrow_pos.y - avoider_pos.y, arrow_pos.x - avoider_pos.x
            )
            # get the index of the cone
            cone_index = int((angle + np.pi) / (2 * np.pi) * self.cones)
            # get the distance to the arrow
            distance = 1 - (arrow_pos - avoider_pos).length / np.sqrt(VIEWPORT_W ** 2 + VIEWPORT_H ** 2)
            # calculate cosine of the angle between the arrow velocity and the avoider velocity the closer the angle is to 180 degrees the more dangerous the arrow is
            arrow_vel = arrow.linearVelocity
            avoider_vel = self.avoider.linearVelocity
            danger = np.dot(arrow_vel, avoider_vel) / (np.linalg.norm(arrow_vel) * np.linalg.norm(avoider_vel))
            if distance > arrow_distances[cone_index]:
                arrow_distances[cone_index] = distance
                arrow_directions[cone_index] = danger

        # now add the arrow distances to the observation
        self_observation = np.concatenate((self_observation, arrow_distances, arrow_directions))
        return self_observation

    def calculate_reward(self, obs: np.array):
        # we want to reward the agent for being in the center of the screen
        # we also want to punish the agent for being close to the arrows

        # reward for being in the center of the screen
        reward = 0
        # reward += CENTER_REWARD * (2 - abs(obs[0]) - abs(obs[1]))

        # reward for being far from the arrows
        # for i in range(4, 4 + self.cones):
        #     reward += ARROW_REWARD * (1 - obs[i])

        # reward for being alive is constant
        reward += ALIVE_REWARD
        return reward



    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # shot arrows
        self.shoot_arrow()

        # Apply force to the avoider
        if action == 1:
            self.avoider.ApplyForceToCenter((ENGINE_POWER, 0), True)
        elif action == 2:
            self.avoider.ApplyForceToCenter((-ENGINE_POWER, 0), True)
        elif action == 3:
            self.avoider.ApplyForceToCenter((0, ENGINE_POWER), True)
        elif action == 4:
            self.avoider.ApplyForceToCenter((0, -ENGINE_POWER), True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # handle arrows
        self.update_arrows()

        # if game over, return observation, reward, done, truncated, info
        if self.game_over:
            return np.array(0, dtype=np.float32), self.reward, self.game_over, False, {}

        obs = self.get_observation()

        self.steps += 1
        reward = self.calculate_reward(obs)
        return np.array(obs, dtype=np.float32), reward, self.game_over, False, {}



    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (0, 0, 0), self.surf.get_rect())

        # draw avoider onto surf
        pygame.draw.circle(self.surf, (0, 0, 255), (self.avoider.position.x, self.avoider.position.y), AVOIDER_RADIUS)

        # draw arrows onto surf
        for arrow in self.arrows:
            pygame.draw.circle(self.surf, (255, 0, 0), (arrow.position.x, arrow.position.y), ARROW_RADIUS)

        # draw walls onto surf
        pygame.draw.rect(self.surf, (255, 255, 255), (self.walls[0].position.x, self.walls[0].position.y, 3, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), (self.walls[1].position.x-2, self.walls[1].position.y + 2, 3, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), (self.walls[2].position.x, self.walls[2].position.y, VIEWPORT_W, 3))
        pygame.draw.rect(self.surf, (255, 255, 255), (self.walls[3].position.x, self.walls[3].position.y - 2, VIEWPORT_W, 3))


        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.isopen = False
