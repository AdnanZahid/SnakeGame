# Imports
import pygame
from random import *

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
snake_color = (0, 255, 0)
food_color = (0, 0, 255)

# Grid constants
columns, rows = screen_size[0], screen_size[1];

# Snake constants
snake_initial_size = 1
snake_position = (10,10)

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
        return (left == NodeType.wall,forward == NodeType.wall,right == NodeType.wall)

def neuralInputs(snake_nodes,direction,grid):
        return (areNeighboringNodesBlocked(*getNeighboringNodes(snake_nodes,direction,grid)),random())

def runGame():

        # count = 0

        # Game objects
        direction = Direction.right
        grid = getGrid()
        snake_nodes = getSnakeNodes(snake_position[0],
                                    snake_position[1],
                                    grid)
        pygame.init()
        screen = pygame.display.set_mode((screen_size[0]*block_size,
                                          screen_size[1]*block_size))

        # Game loop
        while not isGameOver(snake_nodes):

                # count = count + 1

                print neuralInputs(snake_nodes,direction,grid)

                for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                                game_over = True

                # Controls
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_UP] and direction!=Direction.down: direction = Direction.up
                elif pressed[pygame.K_DOWN] and direction!=Direction.up: direction = Direction.down
                elif pressed[pygame.K_LEFT] and direction!=Direction.right: direction = Direction.left
                elif pressed[pygame.K_RIGHT] and direction!=Direction.left: direction = Direction.right

                # Drawing
                screen.fill(screen_color)
                drawNodes(grid,screen)
                pygame.display.flip()

                # Clock ticking
                pygame.time.Clock().tick(60)

                # if count % 5 == 0:
                snake_nodes = advanceSnake(snake_nodes,direction,grid)
        runGame()

runGame()