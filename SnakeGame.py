# Imports
import pygame
from random import randint
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv

# Enums
class Direction:
    left, right, up, down = range(4)
class NodeType:
    empty, snake, food, wall = range(4)

# Screen constants
block_size = 10
screen_size = (50,50)
screen_color = (0, 0, 0)
wall_color = (128, 128, 128)
snake_color = (0, 255, 255)
food_color = (0, 255, 0)

# Grid constants
columns, rows = screen_size[0], screen_size[1];

# Snake constants
snake_initial_size = 1
snake_position = (randint(1, columns-snake_initial_size),randint(1, rows-snake_initial_size))

class SnakeNode:
        def __init__(self,x,y):
                self.x = x
                self.y = y

def getGrid():
        grid = [[0 for x in range(columns)] for y in range(rows)]

        for x in range(columns):
                grid[x][0] = NodeType.wall
                grid[x][columns-1] = NodeType.wall

        for y in range(rows):
                grid[0][y] = NodeType.wall
                grid[rows-1][y] = NodeType.wall

        return grid

def getSnakeNodes(x,y,grid):
        # Create initial snake
        snake_nodes = []
        for i in range(snake_initial_size):
            segment = SnakeNode(x+i, y)
            snake_nodes.append(segment)
            grid[x+i][y] = NodeType.snake

        return snake_nodes

def drawNode(x,y,grid,screen):
        if grid[x][y] == NodeType.snake:  color = snake_color
        elif grid[x][y] == NodeType.food: color = food_color
        elif grid[x][y] == NodeType.wall: color = wall_color
        else:                             color = screen_color

        pygame.draw.rect(screen,color,pygame.Rect(x*block_size,y*block_size,block_size,block_size))

def isGameOver(snake_nodes):
        head = snake_nodes[0]
        return head.x == 0\
                or head.y == 0\
                or head.x == columns-1\
                or head.y == rows-1

def advanceSnake(snake_nodes,direction,grid):
        head = snake_nodes[0]
        tail = snake_nodes.pop()
        grid[tail.x][tail.y] = NodeType.empty
        
        if direction == Direction.up:
                tail.x = head.x
                tail.y = head.y - 1
        elif direction == Direction.down:
                tail.x = head.x
                tail.y = head.y + 1
        elif direction == Direction.left:
                tail.x = head.x - 1
                tail.y = head.y
        elif direction == Direction.right:
                tail.x = head.x + 1
                tail.y = head.y

        snake_nodes.insert(0,tail)
        grid[tail.x][tail.y] = NodeType.snake
        return snake_nodes

def drawNodes(grid,screen):
        for x in range(columns):
                for y in range(rows):
                        drawNode(x,y, grid,screen)

def getNeighboringNodes(snake_nodes,direction,grid): # Left, forward, right nodes of snake
        head = snake_nodes[0]

        if direction == Direction.right:
                return (grid[head.x][head.y-1],grid[head.x+1][head.y],grid[head.x][head.y+1])
        elif direction == Direction.left:
                return (grid[head.x][head.y+1],grid[head.x-1][head.y],grid[head.x][head.y-1])
        elif direction == Direction.up:
                return (grid[head.x-1][head.y],grid[head.x][head.y-1],grid[head.x+1][head.y])
        else:
                return (grid[head.x+1][head.y],grid[head.x][head.y+1],grid[head.x-1][head.y])

def areNeighboringNodesBlocked(left,forward,right):
        return (int(left == NodeType.wall),int(forward == NodeType.wall),int(right == NodeType.wall))

def isAnyNeighboringNodesBlocked(left,forward,right):
        return left == NodeType.wall or forward == NodeType.wall or right == NodeType.wall

def neuralInputs(snake_nodes,grid,absolute_direction,relative_direction):
        return (areNeighboringNodesBlocked(*getNeighboringNodes(snake_nodes,absolute_direction,grid)),relative_direction)

def getTrainedModel(data, labels):
    network = input_data(shape=[None, 4, 1], name="input")
    network = fully_connected(network, 1, activation="linear")
    network = regression(network, optimizer="adam", learning_rate=1e-2, loss="mean_square", name="target")
    model = tflearn.DNN(network)

    data = np.array(data).reshape(-1, 4, 1)
    labels = np.array(labels).reshape(-1, 1)

    model.fit(data, labels, n_epoch = 10, shuffle = True)
    return model

def getPredictedDirection(snake_nodes,absolute_direction,relative_direction,model,inputs,grid):
    head = snake_nodes[0]

    # print(model.predict(preprocess(inputs)))

    prediction = model.predict(preprocess(inputs))

    if prediction < 0.45:
        relative_direction = randint(-1, 1)
        print "Error # 0"
        return getPredictedDirection(snake_nodes,absolute_direction,relative_direction,model,inputs,grid)
    else:

        # Error debugging
        left,forward,right = getNeighboringNodes(snake_nodes,absolute_direction,grid)
        if left == 0 and forward == NodeType.wall and right == 0 and relative_direction == 0:
            print "Error # 1: "+str(prediction)

        if absolute_direction == Direction.right:
                if relative_direction == -1:  return Direction.up,relative_direction
                elif relative_direction == 0: return Direction.right,relative_direction
                else:                         return Direction.down,relative_direction
        elif absolute_direction == Direction.left:
                if relative_direction == -1:  return Direction.down,relative_direction
                elif relative_direction == 0: return Direction.left,relative_direction
                else:                         return Direction.up,relative_direction
        elif absolute_direction == Direction.up:
                if relative_direction == -1:  return Direction.left,relative_direction
                elif relative_direction == 0: return Direction.up,relative_direction
                else:                         return Direction.right,relative_direction
        else:
                if relative_direction == -1:  return Direction.right,relative_direction
                elif relative_direction == 0: return Direction.down,relative_direction
                else:                         return Direction.left,relative_direction

def preprocess(data):
    for i in range(len(data)):
        return np.array(data, dtype=np.float32).reshape(-1, 4, 1)

def getInputs(snake_nodes,grid,absolute_direction,relative_direction):
    inputs = neuralInputs(snake_nodes,grid,absolute_direction,relative_direction)
    return inputs[0][0],inputs[0][1],inputs[0][2],inputs[1]

def getOutputForTraining(inputs,snake_nodes,direction,grid):
    return "\n{},{},{},{},{}".format(int(not isGameOver(snake_nodes)),*inputs)

def runGame(death_count,font,model):

        # Game objects
        direction = Direction.right
        grid = getGrid()
        snake_nodes = getSnakeNodes(snake_position[0],
                                    snake_position[1],
                                    grid)
        screen = pygame.display.set_mode((screen_size[0]*block_size,
                                          screen_size[1]*block_size))
        death_count += 1

        # Game loop
        while not isGameOver(snake_nodes):

                # Update score
                death_count_label = font.render("Death count: {}".format(death_count), 1, (255,255,0))

                for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                                game_over = True

                # Drawing
                screen.fill(screen_color)
                drawNodes(grid,screen)
                screen.blit(death_count_label, (0, 0))
                pygame.display.flip()

                # Clock ticking
                pygame.time.Clock().tick(9999999999999)

                # Manual controls
                # pressed = pygame.key.get_pressed()
                # if pressed[pygame.K_UP] and direction!=Direction.down: direction = Direction.up
                # elif pressed[pygame.K_DOWN] and direction!=Direction.up: direction = Direction.down
                # elif pressed[pygame.K_LEFT] and direction!=Direction.right: direction = Direction.left
                # elif pressed[pygame.K_RIGHT] and direction!=Direction.left: direction = Direction.right

                # Random controls
                # pressed = randint(0, 3)
                # if pressed == 0 and direction!=Direction.down: direction = Direction.up
                # elif pressed == 1 and direction!=Direction.up: direction = Direction.down
                # elif pressed == 2 and direction!=Direction.right: direction = Direction.left
                # elif pressed == 3 and direction!=Direction.left: direction = Direction.right

                # AI controls
                relative_direction = randint(-1, 1)
                inputs = getInputs(snake_nodes,grid,direction,relative_direction)

                if isAnyNeighboringNodesBlocked(*getNeighboringNodes(snake_nodes,direction,grid)) == True:
                    direction,relative_direction = getPredictedDirection(snake_nodes,direction,relative_direction,model,inputs,grid)

                    snake_nodes = advanceSnake(snake_nodes,direction,grid)
                    output = getOutputForTraining(inputs,snake_nodes,direction,grid)
                    # print(output)
                    file = open("Data.csv","a")
                    file.write(output)
                    file.close()

                else:
                    snake_nodes = advanceSnake(snake_nodes,direction,grid)
        runGame(death_count,font,model)

data,labels = load_csv("Data.csv",target_column=0,categorical_labels=True,n_classes=2)
model = getTrainedModel(data,labels)
death_count = 0
pygame.init()
font = pygame.font.SysFont("monospace", 50)
runGame(death_count,font,model)