# Credit to Cameron Jackson for starting this project and https://github.com/lia-univali/neat-snake/blob/master/game.py for being a reference
import pygame
import time
import random
import math
import neat
import sys
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def find_nearest(x_change, y_change, snake_positions, fruit_x, fruit_y):
    x_curr = snake_positions[-1][0] + x_change
    y_curr = snake_positions[-1][1] + y_change
    while not(x_curr > 999 or x_curr < 1 or y_curr > 999 or y_curr < 1):
        for xpos, ypos in snake_positions[:-1]:
            if xpos == x_curr and ypos == y_curr:
                return math.sqrt((x_curr-snake_positions[-1][0])**2 + (y_curr-snake_positions[-1][1])**2)
        if x_curr == fruit_x and y_curr == fruit_y:
            return math.sqrt((x_curr-snake_positions[-1][0])**2 + (y_curr-snake_positions[-1][1])**2)
        x_curr += x_change
        y_curr += y_change
    return math.sqrt((x_curr-snake_positions[-1][0])**2 + (y_curr-snake_positions[-1][1])**2)


def draw_fruit(snakePositions):
    on_snake = True
    while on_snake:
        on_snake = False
        fruitx = random.randint(0, 49) * 20
        fruity = random.randint(0, 49) * 20
        for pos in snakePositions:
            if pos[0] - 10 == fruitx and pos[1] - 10 == fruity:
                on_snake = True

    return fruitx, fruity


def play_game(genome, config):
    agent = neat.nn.FeedForwardNetwork.create(genome, config)
    x = 500
    y = 500
    snake_positions = [(x, y)]
    x_direction = 20
    y_direction = 0
    speed = 200
    score = 0
    pygame.init()
    frames_alive = 0
    previous_direction = 0
    fruitx, fruity = draw_fruit(snake_positions)

    playing = True
    while playing:
        # ML variables
        dist_to_wall = [x, y, 1000 - x, 1000 - y]
        score = score
        frt = [fruitx, fruity]
        nearest = [
            find_nearest(-20, 0, snake_positions, fruitx, fruity),
            find_nearest(-20, -20, snake_positions, fruitx, fruity),
            find_nearest(0, -20, snake_positions, fruitx, fruity),
            find_nearest(20, -20, snake_positions, fruitx, fruity),
            find_nearest(20, 0, snake_positions, fruitx, fruity),
            find_nearest(20, 20, snake_positions, fruitx, fruity),
            find_nearest(0, 20, snake_positions, fruitx, fruity),
            find_nearest(-20, 20, snake_positions, fruitx, fruity)
        ]

        info_vector = []
        for info in nearest:
            info_vector.append(info)
        for info in dist_to_wall:
            info_vector.append(info)
        for info in frt:
            info_vector.append(info)

        # blank the screen

        # update snake positions
        snake_positions.pop(0)
        x += x_direction
        y += y_direction
        snake_positions.append((x, y))
        # draw snake

        outputs = agent.activate(info_vector)
        direction = outputs.index(max(outputs))

        if direction == 0 and not x_direction == 20:  # left
            x_direction = -20
            y_direction = 0

        if direction == 1 and not x_direction == -20:
            x_direction = 20
            y_direction = 0

        if direction == 2 and not y_direction == -20:  # turn down
            x_direction = 0
            y_direction = 20

        if direction == 3 and not y_direction == 20:  # turn down
            x_direction = 0
            y_direction = -20

        previous_direction = direction

        for pos in snake_positions[0:-1]:
            if x == pos[0] and y == pos[1]:
                # end the game if snakes runs into itself
                playing = False

        if (x > fruitx - 10 and x < fruitx + 10):
            if (y > fruity - 10 and y < fruity + 10):
                # snake goes over fruit
                # draw over rectangle
                score = score + 100
                snake_positions.append(snake_positions[-1])
                fruitx, fruity = draw_fruit(snake_positions)

        if x > 1000 or x < 0 or y < 0 or y > 1000:
            playing = False
        frames_alive += 1

    return (0.5*frames_alive) * score


def eval_genome(genome, config):
    genome.fitness = play_game(genome, config)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)


path_to_cfg = "./config"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     path_to_cfg)

# Create the population
p = neat.Population(config)
if len(sys.argv) > 1:
    p = load_object(sys.argv[1])
    print("Starting from Checkpoint at " + sys.argv[1])
# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1000))

winner = p.run(eval_genomes, 10000)
