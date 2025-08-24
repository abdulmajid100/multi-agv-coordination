import pygame
#import time
import random

# Initialize pygame
pygame.init()

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Set display dimensions
display_width = 600
display_height = 400

# Create display
game_display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Snake Game')

# Set game clock
clock = pygame.time.Clock()

# Set snake block size and speed
snake_block = 10
snake_speed = 15

# Set fonts
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

def your_score(score):
    """Display the current score on the screen"""
    value = score_font.render("Your Score: " + str(score), True, white)
    game_display.blit(value, [0, 0])

def our_snake(snake_block, snake_list):
    """Draw the snake on the screen"""
    for x in snake_list:
        pygame.draw.rect(game_display, green, [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    """Display a message on the screen"""
    mesg = font_style.render(msg, True, color)
    game_display.blit(mesg, [display_width / 6, display_height / 3])

def game_loop():
    """Main game loop"""
    game_over = False
    game_close = False

    # Initial snake position
    x1 = display_width / 2
    y1 = display_height / 2

    # Initial movement direction
    x1_change = 0
    y1_change = 0

    # Initialize snake
    snake_list = []
    length_of_snake = 1

    # Generate initial food position
    foodx = round(random.randrange(0, display_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, display_height - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close:
            # Game over screen
            game_display.fill(black)
            message("You Lost! Press Q-Quit or C-Play Again", red)
            your_score(length_of_snake - 1)
            pygame.display.update()

            # Handle game over input
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        # Handle game input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and x1_change == 0:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT and x1_change == 0:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP and y1_change == 0:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN and y1_change == 0:
                    y1_change = snake_block
                    x1_change = 0

        # Update snake position
        x1 += x1_change
        y1 += y1_change

        # Wrap around screen edges instead of collision
        if x1 >= display_width:
            x1 = 0
        elif x1 < 0:
            x1 = display_width - snake_block

        if y1 >= display_height:
            y1 = 0
        elif y1 < 0:
            y1 = display_height - snake_block

        # Draw game elements
        game_display.fill(black)
        pygame.draw.rect(game_display, red, [foodx, foody, snake_block, snake_block])

        # Update snake
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)

        # Remove extra snake segments
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        # Check for self collision
        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True

        # Draw snake and score
        our_snake(snake_block, snake_list)
        your_score(length_of_snake - 1)

        pygame.display.update()

        # Check if food eaten
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, display_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, display_height - snake_block) / 10.0) * 10.0
            length_of_snake += 1

        # Control game speed
        clock.tick(snake_speed)

    # Quit pygame
    pygame.quit()
    quit()

# Start the game
game_loop()
