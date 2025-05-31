import pygame
import heapq
import time
import math
import random

pygame.init()

CELL_SIZE = 30
GRID_WIDTH = 19
GRID_HEIGHT = 21
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE + 50 
BLACK = (0, 0, 0)
lightgreen = (144, 238, 144)
gray = (128, 128, 128)
beige = (245, 245, 220)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 182, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 182, 85)
BLUE = (33, 33, 255)
Brown=(196, 164, 132)

class PacmanGame:
    def __init__(self, mode):
        self.mode = mode
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman Game")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.grid = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,3,1],
            [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
            [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
            [1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1],
            [1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1],
            [1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1],
            [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
            [1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1],
            [1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1],
            [1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1],
            [1,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,1],
            [1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ]
        
        self.pacman_pos = [9, 15] 
        self.pacman_direction = 0  
        self.pacman_anim_frame = 0
        self.pacman_mouth_angle = 0 
        self.mouth_open = True
        self.ghost_start_positions = [
            [9, 8],
            [9, 11],]
        self.ghosts = [
            {"pos": [8, 9], "color": PINK, "speed": 0.5, "direction": 0, "eyes_offset": 0, "frightened": False},
            {"pos": [9, 11], "color": ORANGE, "speed": 0.7, "direction": 0, "eyes_offset": 0, "frightened": False}
        ]
        
        self.ghost_eaten = [False] * len(self.ghosts)
        self.ghost_original_colors = [ PINK,  ORANGE]
        
        self.score = 0
        self.lives = 3
        self.dots_left = sum(row.count(0) for row in self.grid) + sum(row.count(3) for row in self.grid)
        self.last_ghost_move = time.time()
        self.ghost_move_delay = 0.3 
        self.last_anim_update = time.time()
        self.anim_delay = 0.15
        
        self.power_mode = False
        self.power_timer = 0
        self.power_duration = 9.0 
        self.ghost_points = 200 
        self.ghosts_eaten_count = 0  
    
        self.pellet_pulse = 0
        self.pellet_growing = True
        
        self.particles = []
        
        self.score_popups = []
        
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  

        self.direction_angles = [0, 90, 180, 270]  
        
        self.last_ai_move_time = time.time()
        self.ai_move_delay = 0.2  
        self.last_manual_move_time = time.time()
        self.manual_move_delay = 0.08  

        
    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return dx + dy
    
    def get_neighbors(self, pos):
        neighbors = []
        for dx, dy in self.directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if (0 <= new_x < GRID_WIDTH and 
                0 <= new_y < GRID_HEIGHT and 
                self.grid[new_y][new_x] != 1):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def greedy_search(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (self.heuristic(start, goal), start))
        came_from = {start: None}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    priority = self.heuristic(neighbor, goal)  
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        
        path = []
        current = goal
        while current != start:
            if current not in came_from:
                return []  
            path.append(current)
            current = came_from[current]
        
        path.reverse()
        return path


    def get_opposite_direction(self, direction):
        return (direction + 2) % 4
    
    def move_ghosts(self):
        current_time = time.time()
        for i, ghost in enumerate(self.ghosts):
            if current_time - ghost.get("last_move", 0) >= ghost["speed"]:
                if self.ghost_eaten[i]:
                    continue
                
                start = tuple(ghost["pos"])
                neighbors = self.get_neighbors(start)
                
                opposite_dir = self.get_opposite_direction(ghost["direction"])
                valid_neighbors = [
                    n for n in neighbors 
                    if self.directions.index((n[0]-start[0], n[1]-start[1])) != opposite_dir
                ]
                
                if not valid_neighbors:
                    valid_neighbors = neighbors.copy()
                
                if self.power_mode and ghost["frightened"]:
                    best_dist = -float('inf')
                    best_pos = start
                    for n in valid_neighbors:
                        dx = self.pacman_pos[0] - n[0]
                        dy = self.pacman_pos[1] - n[1]
                        dist = math.hypot(dx, dy)
                        if dist > best_dist:
                            best_dist = dist
                            best_pos = n
                    ghost["pos"] = list(best_pos)
                else:
                    path = self.greedy_search(start, tuple(self.pacman_pos))
                    if path:
                        next_pos = path[0]
                    else:
                        min_dist = float('inf')
                        next_pos = start
                        for n in valid_neighbors:
                            dx = self.pacman_pos[0] - n[0]
                            dy = self.pacman_pos[1] - n[1]
                            dist = abs(dx) + abs(dy)
                            if dist < min_dist:
                                min_dist = dist
                                next_pos = n
                    ghost["pos"] = list(next_pos)
                
                dx = ghost["pos"][0] - start[0]
                dy = ghost["pos"][1] - start[1]
                if dx > 0:
                    ghost["direction"] = 0
                elif dx < 0:
                    ghost["direction"] = 2
                elif dy > 0:
                    ghost["direction"] = 1
                else:
                    ghost["direction"] = 3
                
                ghost["last_move"] = current_time

    def update_animations(self):
       current_time = time.time()
       if current_time - self.last_anim_update >= self.anim_delay:
         self.mouth_open = not self.mouth_open
         self.pacman_anim_frame = (self.pacman_anim_frame + 1) % 4

         self.pellet_pulse += 0.5 if self.pellet_growing else -0.5
         if self.pellet_pulse >= 3:
            self.pellet_growing = False
         elif self.pellet_pulse <= 0:
            self.pellet_growing = True
             
         if self.power_mode and self.power_timer < 3.0:
            flash_color = BLUE if self.pacman_anim_frame % 2 == 0 else WHITE
            for i, ghost in enumerate(self.ghosts):
                if ghost.get("frightened") and not self.ghost_eaten[i]:
                    ghost["color"] = flash_color

         self.last_anim_update = current_time
       self.particles = [p for p in self.particles if p["life"] > 0]
       for p in self.particles:
          p["x"] += p["dx"]
          p["y"] += p["dy"]
          p["life"] -= 1
          p["size"] = max(1, p["size"] * 0.95)
            
    def update_power_mode(self):
      if self.power_mode and time.time() - self.power_timer >= self.power_duration:
        self.power_mode = False
        self.ghosts_eaten_count = 0

        for i, ghost in enumerate(self.ghosts):
            if not self.ghost_eaten[i]:
                ghost["color"] = self.ghost_original_colors[i]
                ghost["frightened"] = False

    def check_collision(self):
        for i, ghost in enumerate(self.ghosts):
            if self.pacman_pos == ghost["pos"]and not self.ghost_eaten[i]:
                if self.power_mode and ghost["frightened"]:
                    ghost_points = self.ghost_points 
                    self.score += ghost_points
                    self.ghosts_eaten_count += 1
                    ghost["pos"] = self.ghost_start_positions[i].copy()
                    self.ghost_eaten[i] = False  
                    ghost["frightened"] = False
                    ghost["color"] = self.ghost_original_colors[i]
                    self.create_particles(ghost["pos"][0] * CELL_SIZE + CELL_SIZE//2, 
                                       ghost["pos"][1] * CELL_SIZE + CELL_SIZE//2,
                                       BLUE, 15)
                    return False  
                else:
                    return True  
        return False

    def create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                "x": x,
                "y": y,
                "dx": math.cos(angle) * speed,
                "dy": math.sin(angle) * speed,
                "color": color,
                "size": random.uniform(2, 5),
                "life": random.randint(10, 30)
            })
            
    def collect_dot(self):
      x, y = self.pacman_pos
      cell = self.grid[y][x]
      if cell in (0, 3): 
        self.grid[y][x] = 2
        self.dots_left -= 1
        if cell == 0:  
            self.score += 5
            color, particles = WHITE, 3
        else:  
            self.score += 20
            color, particles = Brown, 10
            self.power_mode = True
            self.power_timer = time.time()
            self.ghosts_eaten_count = 0
            for i, ghost in enumerate(self.ghosts):
                if not self.ghost_eaten[i]:
                    ghost["color"] = BLUE
                    ghost["frightened"] = True
        
        self.create_particles(x * CELL_SIZE + CELL_SIZE // 2,
                              y * CELL_SIZE + CELL_SIZE // 2,
                              color, particles)

    def respawn_ghost(self, ghost_index):
        start_positions = [[9, 8],  [9, 11]]
        self.ghosts[ghost_index]["pos"] = start_positions[ghost_index].copy()
        self.ghost_eaten[ghost_index] = False
        if self.power_mode:
            self.ghosts[ghost_index]["color"] = BLUE
            self.ghosts[ghost_index]["frightened"] = True
        else:
            self.ghosts[ghost_index]["color"] = self.ghost_original_colors[ghost_index]
            self.ghosts[ghost_index]["frightened"] = False

    def draw_pacman(self):
      x, y = self.pacman_pos
      center_x = x * CELL_SIZE + CELL_SIZE // 2
      center_y = y * CELL_SIZE + CELL_SIZE // 2
      radius = CELL_SIZE // 2 - 2
      mouth_angle = 45 if self.mouth_open else 10
      direction_angle = self.direction_angles[self.pacman_direction]
      start_angle = direction_angle + mouth_angle
      end_angle = direction_angle + 360 - mouth_angle
      pygame.draw.arc(self.screen, YELLOW,
                   (center_x - radius, center_y - radius, radius*2, radius*2),
                   math.radians(start_angle), math.radians(end_angle), radius)

      if mouth_angle > 15:
        for angle in [start_angle, end_angle]:
            x_end = center_x + radius * math.cos(math.radians(angle))
            y_end = center_y - radius * math.sin(math.radians(angle))
            pygame.draw.line(self.screen, YELLOW, (center_x, center_y), (x_end, y_end), 2)

    def draw_ghost(self, ghost, index):
      x, y = ghost["pos"]
      color = ghost["color"]
      direction = ghost["direction"]
      eyes_offset = ghost["eyes_offset"]
      if self.ghost_eaten[index]:
        eye_radius = 4
        eye_y_pos = y * CELL_SIZE + 11 + eyes_offset
        left_eye_x = x * CELL_SIZE + CELL_SIZE // 4
        right_eye_x = x * CELL_SIZE + CELL_SIZE - CELL_SIZE // 4
        
        pygame.draw.circle(self.screen, WHITE, (left_eye_x, eye_y_pos), eye_radius)
        pygame.draw.circle(self.screen, WHITE, (right_eye_x, eye_y_pos), eye_radius)

        pupil_offset = [(2, 0), (0, 2), (-2, 0), (0, -2)]  
        p_dx, p_dy = pupil_offset[direction]
        pygame.draw.circle(self.screen, gray, 
                           (left_eye_x + p_dx, eye_y_pos + p_dy), 2)
        pygame.draw.circle(self.screen, gray, 
                           (right_eye_x + p_dx, eye_y_pos + p_dy), 2)
        return
      left = x * CELL_SIZE + 3
      top = y * CELL_SIZE + 3
      width = CELL_SIZE - 6
      height = CELL_SIZE - 8
      ghost_rect = pygame.Rect(left, top, width, height - 4)
      pygame.draw.rect(self.screen, color, ghost_rect, border_radius=12)
      pygame.draw.ellipse(self.screen, color, 
                           (left, top - 4, width, 10))
      wave_segments = 3
      segment_width = width / wave_segments
        
      for i in range(wave_segments):
            wave_left = left + i * segment_width
            pygame.draw.arc(self.screen, color,
                           (wave_left, top + height - 8, segment_width, 8),
                           math.pi, 2 * math.pi, 4)  
      eye_radius = 4
      eye_y_pos = top + 8 + eyes_offset
      left_eye_x = left + width // 4
      right_eye_x = left + width - width // 4
      pupil_offset = [(2, 0), (0, 2), (-2, 0), (0, -2)]  

      if ghost["frightened"] and self.power_mode:
        pygame.draw.circle(self.screen, WHITE, (left_eye_x, eye_y_pos), eye_radius - 1)
        pygame.draw.circle(self.screen, WHITE, (right_eye_x, eye_y_pos), eye_radius - 1)
      else:
        pygame.draw.circle(self.screen, WHITE, (left_eye_x, eye_y_pos), eye_radius)
        pygame.draw.circle(self.screen, WHITE, (right_eye_x, eye_y_pos), eye_radius)

      p_dx, p_dy = pupil_offset[direction]
      pygame.draw.circle(self.screen, gray, 
                       (left_eye_x + p_dx, eye_y_pos + p_dy), 2)
      pygame.draw.circle(self.screen, gray, 
                       (right_eye_x + p_dx, eye_y_pos + p_dy), 2)
   
    def draw_particles(self):
        for particle in self.particles:
            pygame.draw.circle(self.screen, particle["color"], 
                             (int(particle["x"]), int(particle["y"])), 
                             int(particle["size"]))
    
    def draw_power_timer(self):
        if self.power_mode:
            current_time = time.time()
            elapsed = current_time - self.power_timer
            remaining = max(0, self.power_duration - elapsed)
            remaining_width = (remaining / self.power_duration) * 150
            bar_x = (SCREEN_WIDTH - 150) // 2
            bar_y = SCREEN_HEIGHT - 20
            
            pygame.draw.rect(self.screen, gray, (bar_x, bar_y, 150, 10))
            
            if remaining < 3.0:
                if self.pacman_anim_frame % 2 == 0:
                    color = RED
                else:
                    color = YELLOW
            else:
                color = Brown
                
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, remaining_width, 10))
            
            if self.ghosts_eaten_count > 0:
                next_ghost_value = self.ghost_points 
                value_text = self.small_font.render(f"Next: {next_ghost_value}", True, WHITE)
                self.screen.blit(value_text, (bar_x + 150 + 10, bar_y - 5))
    
    def draw(self):
        self.screen.fill(lightgreen)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, gray, rect)
                    pygame.draw.rect(self.screen, beige, rect.inflate(-4, -4))
                elif self.grid[y][x] == 0:
                    dot_center = (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(self.screen, (180, 180, 180, 128),
                                     dot_center, 5, 1)
                    pygame.draw.circle(self.screen, WHITE, dot_center, 3)
                elif self.grid[y][x] == 3:
                    dot_center = (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(self.screen, (230, 230, 180, 150),
                                     dot_center, 8 + self.pellet_pulse, 1)
                    pygame.draw.circle(self.screen, WHITE, dot_center, 5 + self.pellet_pulse)
        self.update_animations()
        self.draw_particles()
        self.draw_pacman()
        
        for i, ghost in enumerate(self.ghosts):
            self.draw_ghost(ghost, i)
        if self.power_mode:
            self.draw_power_timer()
        
        score_text = self.font.render(f"Score: {self.score}", True, Brown)
        lives_text = self.font.render(f"Lives: {self.lives}", True, YELLOW)
        self.screen.blit(score_text, (10, SCREEN_HEIGHT - 40))
        self.screen.blit(lives_text, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 40))
        pygame.display.flip()
    
    def find_closest_power_pellet(self):
        pac_x, pac_y = self.pacman_pos
        min_dist = float('inf')
        closest_pellet = None
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 3: 
                    dist = abs(x - pac_x) + abs(y - pac_y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pellet = (x, y)
        
        return closest_pellet
    def is_ghost_nearby(self, threshold=5) -> bool:
        pac_x, pac_y = self.pacman_pos
        
        for i, ghost in enumerate(self.ghosts):
            if self.ghost_eaten[i]:
                continue
                
            ghost_x, ghost_y = ghost["pos"]
            dist = abs(ghost_x - pac_x) + abs(ghost_y - pac_y)
            
            if dist <= threshold and not (self.power_mode and ghost["frightened"]):
                return True
        
        return False
    
    def handle_input(self):
        current_time = time.time()
        if current_time - self.last_manual_move_time < self.manual_move_delay:
            return  
        self.last_manual_move_time = current_time
        keys = pygame.key.get_pressed()
        x, y = self.pacman_pos
        moved = False

        requested_direction = None

        if keys[pygame.K_UP]:
            requested_direction = 3  
        elif keys[pygame.K_DOWN]:
            requested_direction = 1  
        elif keys[pygame.K_LEFT]:
            requested_direction = 2 
        elif keys[pygame.K_RIGHT]:
            requested_direction = 0  

        if requested_direction is not None:
            dx, dy = self.directions[requested_direction]
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                if self.grid[new_y][new_x] != 1:  
                    self.pacman_pos = [new_x, new_y]
                    self.pacman_direction = requested_direction
                    moved = True
                else:
                  
                    self.pacman_direction = requested_direction
            else:
               
                self.pacman_direction = requested_direction

       
        if not moved:
            dx, dy = self.directions[self.pacman_direction]
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < GRID_WIDTH and 
                0 <= new_y < GRID_HEIGHT and 
                self.grid[new_y][new_x] != 1):
                self.pacman_pos = [new_x, new_y]
                moved = True

        if moved:
            self.collect_dot()
            self.mouth_open = True


    def ai_move_pacman(self):
        current_time = time.time()
        if current_time - self.last_ai_move_time < self.ai_move_delay:
            return
        self.last_ai_move_time = current_time
        
        start = tuple(self.pacman_pos)
        target = None

        frightened_ghosts = []
        for i, ghost in enumerate(self.ghosts):
            if ghost["frightened"] and not self.ghost_eaten[i]:
                ghost_pos = tuple(ghost["pos"])
                dist = abs(ghost_pos[0] - start[0]) + abs(ghost_pos[1] - start[1])
                if dist <= 3: 
                    frightened_ghosts.append((dist, ghost_pos))
        
        if frightened_ghosts:
            frightened_ghosts.sort()
            target = frightened_ghosts[0][1]
        
        if not target:
            min_dist = float('inf')
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.grid[y][x] in (0, 3):  
                        dist = abs(x - start[0]) + abs(y - start[1])
                        if dist < min_dist:
                            min_dist = dist
                            target = (x, y)
        
        if self.is_ghost_nearby(5) and not self.power_mode:
            closest_pellet = self.find_closest_power_pellet()
            if closest_pellet:
                target = closest_pellet
        
        if target:
            path = self.greedy_search(start, target)
            
            if not path:

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    x = target[0] + dx
                    y = target[1] + dy
                    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                        if self.grid[y][x] != 1:
                            path = self.greedy_search(start, (x, y))
                            if path:
                                break
            
            if path:
                next_pos = path[0]
                dx = next_pos[0] - start[0]
                dy = next_pos[1] - start[1]
                
                if dx > 0: self.pacman_direction = 0
                elif dx < 0: self.pacman_direction = 2
                elif dy > 0: self.pacman_direction = 1
                else: self.pacman_direction = 3
                
                if self.grid[next_pos[1]][next_pos[0]] != 1:
                    self.pacman_pos = list(next_pos)
                    self.collect_dot()
                    self.mouth_open = True
    def run(self):
        clock = pygame.time.Clock()
        game_active = True
        for i in range(4):
            pygame.event.set_allowed(pygame.USEREVENT + i)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if pygame.USEREVENT <= event.type < pygame.USEREVENT + 4:
                    ghost_index = event.type - pygame.USEREVENT
                    self.respawn_ghost(ghost_index)
                
                if not game_active and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.__init__("astar")
                        game_active = True
                    elif event.key == pygame.K_u:
                        self.__init__("user")
                        game_active = True
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return
            
            if game_active:
                if self.mode == "astar":
                    self.ai_move_pacman()
                else:
                    self.handle_input()
                
                self.move_ghosts()
                self.update_power_mode()
                
                if self.check_collision():
                    self.lives -= 1
                    if self.lives > 0:
                        self.pacman_pos = [9, 15]
                        self.pacman_direction = 0
                        
                        for i, ghost in enumerate(self.ghosts):
                            self.respawn_ghost(i)
                            
                        if not self.power_mode:
                            for i, ghost in enumerate(self.ghosts):
                                ghost["color"] = self.ghost_original_colors[i]
                                ghost["frightened"] = False
                    else:
                        game_active = False
                
                if self.dots_left == 0:
                    game_active = False
            
            self.draw()
            
            if not game_active:
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128))
                self.screen.blit(overlay, (0, 0))
                if self.lives <= 0:
                    text = "Game Over! Press A or U to Restart or Q to Quit"
                else:
                    text = "You Win! Press A or U to Play Again or Q to Quit"
                
                game_over_text = self.font.render(text, True, YELLOW)
                text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                
                shadow_text = self.font.render(text, True, BLACK)
                shadow_rect = shadow_text.get_rect(center=(SCREEN_WIDTH//2 + 2, SCREEN_HEIGHT//2 + 2))
                self.screen.blit(shadow_text, shadow_rect)
                self.screen.blit(game_over_text, text_rect)
                
                pygame.display.flip()
            clock.tick(15)  

if __name__ == "__main__":
    game = PacmanGame("astar")
    game.run()