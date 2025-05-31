import pygame
from pacman import PacmanGame

WIDTH, HEIGHT = 600, 600

def draw_start_screen(screen, background_image):
    font = pygame.font.Font(None, 65)
    title_text = font.render("Pacman Game", True, (255, 255, 255))

    font = pygame.font.Font(None, 38)
    supervisor_text_line1 = font.render("Dr. Sara El-Sayed El-Metwally", True, (255, 255, 255))
    supervisor_text_line2 = font.render("Eng. Habiba Mohamed", True, (255, 255, 255))
    instructions_text_line1 = font.render("Press 'A' for A* Algorithm", True, (255, 255, 255))
    instructions_text_line2 = font.render("Press 'U' for User Play", True, (255, 255, 255))

    highlight_color = (0, 205, 139)
    highlight_color2 = (0, 0, 18)

    screen.blit(background_image, (0, 0))

    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    supervisor_rect_line1 = supervisor_text_line1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 10))
    supervisor_rect_line2 = supervisor_text_line2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
    instructions_rect_line1 = instructions_text_line1.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 90))
    instructions_rect_line2 = instructions_text_line2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 130))

    pygame.draw.rect(screen, highlight_color, title_rect.inflate(20, 10))
    pygame.draw.rect(screen, highlight_color2, supervisor_rect_line1.inflate(20, 10))
    pygame.draw.rect(screen, highlight_color2, supervisor_rect_line2.inflate(20, 10))
    pygame.draw.rect(screen, highlight_color2, instructions_rect_line1.inflate(20, 10))
    pygame.draw.rect(screen, highlight_color2, instructions_rect_line2.inflate(20, 10))

    screen.blit(title_text, title_rect)
    screen.blit(supervisor_text_line1, supervisor_rect_line1)
    screen.blit(supervisor_text_line2, supervisor_rect_line2)
    screen.blit(instructions_text_line1, instructions_rect_line1)
    screen.blit(instructions_text_line2, instructions_rect_line2)
    


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pacman Game - A* and Manual")

    try:
        background_image = pygame.image.load("pacman.jpg")
        background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
    except pygame.error as e:
        print(f"Error loading background image: {e}")
        background_image = pygame.Surface((WIDTH, HEIGHT))
        background_image.fill((0, 0, 0)) 

    running = True
    game = None

    while running:
        screen.fill((0, 0, 0))
        if game:
            game.run()
        else:
            draw_start_screen(screen, background_image)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    game = PacmanGame("astar")
                    print("A* algorithm selected.")
                elif event.key == pygame.K_u:
                    game = PacmanGame("user")
                    print("User play selected.")

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
