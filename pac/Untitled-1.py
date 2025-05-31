import os
import sys
import pygame
import numpy as np
import json
import time
from typing import List, Tuple, Dict, Set, Optional, Callable
import heapq

class Config:
    """Configuration settings for the Warehouse Optimizer application."""
    
    # Window settings
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 800
    FPS = 60
    
    # Grid settings
    DEFAULT_GRID_WIDTH = 30
    DEFAULT_GRID_HEIGHT = 20
    MIN_GRID_SIZE = 10
    MAX_GRID_SIZE = 100
    DEFAULT_CELL_SIZE = 30
    MIN_CELL_SIZE = 10
    MAX_CELL_SIZE = 60
    
    # UI settings
    SIDEBAR_WIDTH = 300
    TOOLBAR_HEIGHT = 50
    
    # A* algorithm settings
    DEFAULT_HEURISTIC = "manhattan"
    ANIMATION_SPEED = 0.05  # seconds per step
    
    # Colors
    COLORS = {
        # Main colors
        "background": (24, 24, 27),  # Dark background
        "grid_lines": (75, 85, 99),  # Subtle grid lines
        "text": (241, 245, 249),     # Light text
        "text_secondary": (148, 163, 184),  # Subdued text
        
        # UI elements
        "sidebar": (39, 39, 42),      # Slightly lighter than background
        "toolbar": (39, 39, 42),      # Same as sidebar
        "button": (59, 130, 246),     # Primary blue
        "button_hover": (96, 165, 250),  # Lighter blue
        "button_text": (255, 255, 255),  # White text
        "input_bg": (39, 39, 42),     # Input background
        "input_border": (75, 85, 99), # Input border
        "input_active": (59, 130, 246),  # Active input
        
        # Grid elements
        "empty": (39, 39, 42),        # Empty cell
        "wall": (71, 85, 105),        # Wall/obstacle
        "start": (34, 197, 94),       # Start point (green)
        "end": (239, 68, 68),         # End point (red)
        "path": (250, 204, 21),       # Final path (yellow)
        "visited": (125, 211, 252),   # Visited cells
        "frontier": (147, 51, 234),   # Frontier cells
        
        # Feedback states
        "success": (34, 197, 94),     # Success green
        "warning": (251, 146, 60),    # Warning orange
        "error": (239, 68, 68),       # Error red
        "info": (59, 130, 246),       # Info blue
    }
    
    # Font settings
    FONT_SIZES = {
        "small": 14,
        "medium": 18,
        "large": 24,
        "title": 32
    }

class Warehouse:
    """Model representing a warehouse grid layout."""
    
    def __init__(self, width: int = 30, height: int = 20):
        """
        Initialize a new warehouse grid.
        
        Args:
            width: Width of the grid
            height: Height of the grid
        """
        self.width = width
        self.height = height
        
        # Initialize grid: 0 = empty, 1 = wall
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # Start and end positions
        self.start_pos: Optional[Tuple[int, int]] = None
        self.end_pos: Optional[Tuple[int, int]] = None
        
        # Metadata
        self.name = "New Warehouse"
        self.description = ""
        self.modified = False
    
    def clear(self):
        """Clear the warehouse grid."""
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.start_pos = None
        self.end_pos = None
        self.modified = True
    
    def resize(self, width: int, height: int):
        """
        Resize the warehouse grid.
        
        Args:
            width: New width
            height: New height
        """
        # Create a new grid of the specified size
        new_grid = np.zeros((height, width), dtype=np.int8)
        
        # Copy the old grid data, up to the smaller of the old/new dimensions
        h = min(self.height, height)
        w = min(self.width, width)
        new_grid[:h, :w] = self.grid[:h, :w]
        
        # Update grid and dimensions
        self.grid = new_grid
        self.width = width
        self.height = height
        
        # Check if start and end positions are still valid
        if self.start_pos:
            if (self.start_pos[0] >= height or self.start_pos[1] >= width):
                self.start_pos = None
        
        if self.end_pos:
            if (self.end_pos[0] >= height or self.end_pos[1] >= width):
                self.end_pos = None
        
        self.modified = True
    
    def is_wall(self, row: int, col: int) -> bool:
        """
        Check if a cell is a wall.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if the cell is a wall, False otherwise
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col] == 1
        return False
    
    def set_wall(self, row: int, col: int, is_wall: bool = True):
        """
        Set a cell as a wall or empty.
        
        Args:
            row: Row index
            col: Column index
            is_wall: True to set as wall, False to set as empty
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            # Don't allow walls at start or end positions
            if (row, col) == self.start_pos or (row, col) == self.end_pos:
                return
            
            self.grid[row, col] = 1 if is_wall else 0
            self.modified = True
    
    def toggle_wall(self, row: int, col: int):
        """
        Toggle a cell between wall and empty.
        
        Args:
            row: Row index
            col: Column index
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            # Don't allow walls at start or end positions
            if (row, col) == self.start_pos or (row, col) == self.end_pos:
                return
            
            self.grid[row, col] = 1 - self.grid[row, col]
            self.modified = True
    
    def set_start(self, row: int, col: int):
        """
        Set the start position.
        
        Args:
            row: Row index
            col: Column index
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            # Don't allow start position on a wall
            if self.grid[row, col] == 1:
                return
            
            # Don't allow start position at the end position
            if (row, col) == self.end_pos:
                return
            
            self.start_pos = (row, col)
            self.modified = True
    
    def set_end(self, row: int, col: int):
        """
        Set the end position.
        
        Args:
            row: Row index
            col: Column index
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            # Don't allow end position on a wall
            if self.grid[row, col] == 1:
                return
            
            # Don't allow end position at the start position
            if (row, col) == self.start_pos:
                return
            
            self.end_pos = (row, col)
            self.modified = True
    
    def save(self, filepath: str) -> bool:
        """
        Save the warehouse to a JSON file.
        
        Args:
            filepath: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "width": self.width,
                "height": self.height,
                "grid": self.grid.tolist(),
                "start_pos": self.start_pos,
                "end_pos": self.end_pos,
                "name": self.name,
                "description": self.description
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            self.modified = False
            return True
        except Exception as e:
            print(f"Error saving warehouse: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'Warehouse':
        """
        Load a warehouse from a JSON file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Loaded Warehouse object
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            warehouse = cls(data["width"], data["height"])
            warehouse.grid = np.array(data["grid"], dtype=np.int8)
            warehouse.start_pos = tuple(data["start_pos"]) if data["start_pos"] else None
            warehouse.end_pos = tuple(data["end_pos"]) if data["end_pos"] else None
            warehouse.name = data["name"]
            warehouse.description = data.get("description", "")
            
            return warehouse
        except Exception as e:
            print(f"Error loading warehouse: {e}")
            return cls()  # Return a default warehouse
    
    def generate_random(self, wall_density: float = 0.2):
        """
        Generate a random warehouse layout.
        
        Args:
            wall_density: Probability of a cell being a wall (0.0 to 1.0)
        """
        # Generate random walls
        self.grid = np.random.choice(
            [0, 1], 
            size=(self.height, self.width), 
            p=[1-wall_density, wall_density]
        ).astype(np.int8)
        
        # Clear start and end positions
        self.start_pos = None
        self.end_pos = None
        
        # Find valid positions for start and end
        empty_positions = []
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row, col] == 0:
                    empty_positions.append((row, col))
        
        if len(empty_positions) >= 2:
            import random
            # Set random start and end positions
            start_idx = random.randint(0, len(empty_positions) - 1)
            self.start_pos = empty_positions[start_idx]
            
            # Remove start position from empty positions
            empty_positions.pop(start_idx)
            
            # Set random end position
            end_idx = random.randint(0, len(empty_positions) - 1)
            self.end_pos = empty_positions[end_idx]
        
        self.modified = True

class AStarPathfinder:
    """A* pathfinding algorithm implementation for warehouse optimization."""
    
    def __init__(self, warehouse):
        """
        Initialize the A* pathfinder.
        
        Args:
            warehouse: The warehouse model containing the grid layout
        """
        self.warehouse = warehouse
        
        # Define possible movement directions (including diagonals)
        self.directions = [
            (0, 1),   # right
            (1, 0),   # down
            (0, -1),  # left
            (-1, 0),  # up
            (1, 1),   # down-right
            (1, -1),  # down-left
            (-1, 1),  # up-right
            (-1, -1)  # up-left
        ]
        
        # Heuristic functions
        self.heuristics = {
            "manhattan": self._manhattan_distance,
            "euclidean": self._euclidean_distance,
            "diagonal": self._diagonal_distance
        }
        
        # Default heuristic
        self.current_heuristic = "manhattan"
        
        # For visualization and metrics
        self.path_found = False
        self.visited_cells = []
        self.frontier_cells = []
        self.path = []
        self.search_time = 0
        self.path_length = 0
        self.nodes_expanded = 0
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate the Manhattan distance between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            
        Returns:
            The Manhattan distance
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _euclidean_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            
        Returns:
            The Euclidean distance
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
    def _diagonal_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate the diagonal distance (Chebyshev distance) between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            
        Returns:
            The diagonal distance
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + 0.5 * min(dx, dy)
    
    def set_heuristic(self, heuristic_name: str) -> None:
        """
        Set the heuristic function to use.
        
        Args:
            heuristic_name: Name of the heuristic function
        """
        if heuristic_name in self.heuristics:
            self.current_heuristic = heuristic_name
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  callback: Optional[Callable] = None) -> List[Tuple[int, int]]:
        """
        Find the shortest path from start to goal using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            callback: Optional callback function to update visualization during search
            
        Returns:
            A list of positions forming the path from start to goal
        """
        if not self._is_valid_position(start) or not self._is_valid_position(goal):
            return []
        
        # Reset metrics
        self.path_found = False
        self.visited_cells = []
        self.frontier_cells = []
        self.path = []
        self.nodes_expanded = 0
        
        # Track start time
        start_time = time.time()
        
        # Initialize the open and closed sets
        open_set = []
        closed_set = set()
        
        # Cost from start to current node
        g_score = {start: 0}
        
        # Estimated total cost from start to goal through current node
        f_score = {start: self.heuristics[self.current_heuristic](start, goal)}
        
        # For reconstructing the path
        came_from = {}
        
        # Add the start node to the open set
        heapq.heappush(open_set, (f_score[start], start))
        self.frontier_cells.append(start)
        
        while open_set:
            # Get the node with the lowest f_score
            _, current = heapq.heappop(open_set)
            
            # If we've reached the goal, construct the path and return
            if current == goal:
                self.path_found = True
                self.path = self._reconstruct_path(came_from, current)
                self.path_length = len(self.path)
                self.search_time = time.time() - start_time
                return self.path
            
            # Move current from open set to closed set
            closed_set.add(current)
            self.visited_cells.append(current)
            self.frontier_cells.remove(current)
            self.nodes_expanded += 1
            
            # Update visualization if callback is provided
            if callback:
                callback()
            
            # Check all neighbors
            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                # Skip if the neighbor is not valid or is in the closed set
                if (not self._is_valid_position(neighbor) or 
                    self.warehouse.is_wall(neighbor[0], neighbor[1]) or
                    neighbor in closed_set):
                    continue
                
                # Calculate the tentative g_score
                # Use 1.0 for cardinal directions, 1.4 for diagonal
                is_diagonal = direction[0] != 0 and direction[1] != 0
                movement_cost = 1.4 if is_diagonal else 1.0
                tentative_g_score = g_score[current] + movement_cost
                
                # If the neighbor is not in the open set or has a better g_score
                if (neighbor not in g_score or 
                    tentative_g_score < g_score[neighbor]):
                    
                    # Update the path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = (tentative_g_score + 
                                        self.heuristics[self.current_heuristic](neighbor, goal))
                    
                    # Add the neighbor to the open set if it's not already there
                    if neighbor not in [n for _, n in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        self.frontier_cells.append(neighbor)
        
        # If we get here, no path was found
        self.search_time = time.time() - start_time
        return []
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from the came_from dictionary.
        
        Args:
            came_from: Dictionary mapping each position to the position it came from
            current: Current position (goal)
            
        Returns:
            The reconstructed path from start to goal
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        
        # Reverse to get path from start to goal
        return total_path[::-1]
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is valid within the warehouse grid.
        
        Args:
            pos: Position to check (row, col)
            
        Returns:
            True if the position is valid, False otherwise
        """
        row, col = pos
        return (0 <= row < self.warehouse.height and 
                0 <= col < self.warehouse.width)
    
    def get_metrics(self) -> Dict:
        """
        Get the metrics of the last pathfinding operation.
        
        Returns:
            Dictionary containing metrics
        """
        return {
            "path_found": self.path_found,
            "path_length": self.path_length,
            "nodes_expanded": self.nodes_expanded,
            "search_time": self.search_time
        }

class Button:
    """Button UI component."""
    
    def __init__(self, text: str, pos: Tuple[int, int], size: Tuple[int, int], 
                action: Callable, color: Optional[Tuple[int, int, int]] = None, 
                is_selected: bool = False):
        """
        Initialize the button.
        
        Args:
            text: Button text
            pos: Position (x, y)
            size: Size (width, height)
            action: Callback function when button is clicked
            color: Button color (optional)
            is_selected: Whether the button is selected (optional)
        """
        self.text = text
        self.pos = pos
        self.size = size
        self.action = action
        self.color = color or Config.COLORS["button"]
        self.is_selected = is_selected
        
        self.rect = pygame.Rect(pos, size)
        self.is_hovered = False
        self.is_pressed = False
        
        # Animation properties
        self.anim_hover = 0.0  # 0.0 to 1.0
        self.anim_press = 0.0  # 0.0 to 1.0
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            True if the event was handled, False otherwise
        """
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            return False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_pressed:
                self.is_pressed = False
                if self.rect.collidepoint(event.pos):
                    self.action()
                return True
        
        return False
    
    def update(self, dt):
        """
        Update the button.
        
        Args:
            dt: Delta time in seconds
        """
        # Update hover animation
        target_hover = 1.0 if self.is_hovered or self.is_pressed else 0.0
        self.anim_hover += (target_hover - self.anim_hover) * min(1.0, dt * 10)
        
        # Update press animation
        target_press = 1.0 if self.is_pressed else 0.0
        self.anim_press += (target_press - self.anim_press) * min(1.0, dt * 15)
        
        # Update rect position (in case parent UI moved)
        self.rect.topleft = self.pos
    
    def draw(self, surface):
        """
        Draw the button.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Calculate button color based on state
        base_color = self.color
        if self.is_selected:
            # Brighten the color if selected
            r, g, b = base_color
            base_color = min(255, r + 40), min(255, g + 40), min(255, b + 40)
        
        hover_color = Config.COLORS["button_hover"]
        
        # Interpolate between base and hover colors
        r = int(base_color[0] * (1 - self.anim_hover) + hover_color[0] * self.anim_hover)
        g = int(base_color[1] * (1 - self.anim_hover) + hover_color[1] * self.anim_hover)
        b = int(base_color[2] * (1 - self.anim_hover) + hover_color[2] * self.anim_hover)
        
        button_color = (r, g, b)
        
        # Apply press effect (darken)
        if self.anim_press > 0:
            r = int(r * (1 - self.anim_press * 0.2))
            g = int(g * (1 - self.anim_press * 0.2))
            b = int(b * (1 - self.anim_press * 0.2))
            button_color = (r, g, b)
        
        # Calculate button rect with press effect
        button_rect = self.rect.copy()
        if self.anim_press > 0:
            press_offset = int(2 * self.anim_press)
            button_rect.y += press_offset
            button_rect.height -= press_offset
        
        # Draw button
        pygame.draw.rect(surface, button_color, button_rect, border_radius=4)
        
        # Draw button border
        border_color = Config.COLORS["grid_lines"]
        if self.is_selected:
            border_color = Config.COLORS["success"]
        pygame.draw.rect(surface, border_color, button_rect, width=1, border_radius=4)
        
        # Draw text
        font = pygame.font.SysFont(None, Config.FONT_SIZES["small"])
        text_surface = font.render(self.text, True, Config.COLORS["button_text"])
        text_rect = text_surface.get_rect(center=button_rect.center)
        surface.blit(text_surface, text_rect)

class Label:
    """Label UI component."""
    
    def __init__(self, text: str, pos: Tuple[int, int], font_size: int, 
                color: Tuple[int, int, int], align: str = "left"):
        """
        Initialize the label.
        
        Args:
            text: Label text
            pos: Position (x, y)
            font_size: Font size
            color: Text color
            align: Text alignment ('left', 'center', or 'right')
        """
        self.text = text
        self.pos = pos
        self.font_size = font_size
        self.color = color
        self.align = align
        
        self.font = pygame.font.SysFont(None, font_size)
        self.text_surface = self.font.render(text, True, color)
        
        # Set rect based on alignment
        self._update_rect()
    
    def set_text(self, text: str):
        """
        Set the label text.
        
        Args:
            text: New text
        """
        if text != self.text:
            self.text = text
            self.text_surface = self.font.render(text, True, self.color)
            self._update_rect()
    
    def _update_rect(self):
        """Update the label rect based on alignment."""
        if self.align == "left":
            self.rect = self.text_surface.get_rect(midleft=self.pos)
        elif self.align == "center":
            self.rect = self.text_surface.get_rect(center=self.pos)
        elif self.align == "right":
            self.rect = self.text_surface.get_rect(midright=self.pos)
    
    def draw(self, surface):
        """
        Draw the label.
        
        Args:
            surface: Pygame surface to draw on
        """
        surface.blit(self.text_surface, self.rect)

class Slider:
    """Slider UI component."""
    
    def __init__(self, pos: Tuple[int, int], size: Tuple[int, int], 
                min_value: float, max_value: float, value: float,
                on_change: Callable[[float], None]):
        """
        Initialize the slider.
        
        Args:
            pos: Position (x, y)
            size: Size (width, height)
            min_value: Minimum value
            max_value: Maximum value
            value: Initial value
            on_change: Callback function when value changes
        """
        self.pos = pos
        self.size = size
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.on_change = on_change
        
        self.rect = pygame.Rect(pos, size)
        self.handle_rect = pygame.Rect(0, 0, 16, size[1] + 8)
        self._update_handle_position()
        
        self.is_dragging = False
    
    def _update_handle_position(self):
        """Update the handle position based on the current value."""
        normalized = (self.value - self.min_value) / (self.max_value - self.min_value)
        handle_x = self.pos[0] + normalized * (self.size[0] - self.handle_rect.width)
        self.handle_rect.topleft = (handle_x, self.pos[1] - 4)
    
    def _value_from_handle_position(self):
        """Calculate the value based on the handle position."""
        normalized = (self.handle_rect.left - self.pos[0]) / (self.size[0] - self.handle_rect.width)
        return self.min_value + normalized * (self.max_value - self.min_value)
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            True if the event was handled, False otherwise
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.handle_rect.collidepoint(event.pos):
                    self.is_dragging = True
                    return True
                elif self.rect.collidepoint(event.pos):
                    # Click directly on the slider bar
                    self.handle_rect.centerx = event.pos[0]
                    self._clamp_handle_position()
                    self.value = self._value_from_handle_position()
                    self.on_change(self.value)
                    return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_dragging:
                self.is_dragging = False
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                self.handle_rect.centerx = event.pos[0]
                self._clamp_handle_position()
                self.value = self._value_from_handle_position()
                self.on_change(self.value)
                return True
        
        return False
    
    def _clamp_handle_position(self):
        """Clamp the handle position to the slider bounds."""
        min_x = self.pos[0]
        max_x = self.pos[0] + self.size[0] - self.handle_rect.width
        self.handle_rect.left = max(min_x, min(self.handle_rect.left, max_x))
    
    def update(self, dt):
        """
        Update the slider.
        
        Args:
            dt: Delta time in seconds
        """
        # Update rect position (in case parent UI moved)
        self.rect.topleft = self.pos
        self._update_handle_position()
    
    def draw(self, surface):
        """
        Draw the slider.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw track
        track_rect = pygame.Rect(
            self.pos[0], 
            self.pos[1] + self.size[1] // 2 - 2,
            self.size[0],
            4
        )
        pygame.draw.rect(surface, Config.COLORS["input_border"], track_rect, border_radius=2)
        
        # Draw filled portion
        filled_width = self.handle_rect.centerx - self.pos[0]
        if filled_width > 0:
            filled_rect = pygame.Rect(
                self.pos[0],
                self.pos[1] + self.size[1] // 2 - 2,
                filled_width,
                4
            )
            pygame.draw.rect(surface, Config.COLORS["button"], filled_rect, border_radius=2)
        
        # Draw handle
        pygame.draw.rect(surface, Config.COLORS["button"], self.handle_rect, border_radius=8)
        pygame.draw.rect(surface, Config.COLORS["input_border"], self.handle_rect, width=1, border_radius=8)

class GridView:
    """Grid view component for rendering the warehouse grid."""
    
    def __init__(self, warehouse, pathfinder):
        """
        Initialize the grid view.
        
        Args:
            warehouse: Warehouse model
            pathfinder: A* pathfinder
        """
        self.warehouse = warehouse
        self.pathfinder = pathfinder
        
        # Grid rendering properties
        self.cell_size = Config.DEFAULT_CELL_SIZE
        self.offset_x = Config.SIDEBAR_WIDTH
        self.offset_y = Config.TOOLBAR_HEIGHT
        
        # Path visualization
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.frontier_cells: Set[Tuple[int, int]] = set()
        self.path_cells: Set[Tuple[int, int]] = set()
    
    def update(self, dt):
        """
        Update the grid view.
        
        Args:
            dt: Delta time in seconds
        """
        # Adjust cell size based on grid and window size
        available_width = Config.WINDOW_WIDTH - Config.SIDEBAR_WIDTH
        available_height = Config.WINDOW_HEIGHT - Config.TOOLBAR_HEIGHT
        
        cell_width = available_width / self.warehouse.width
        cell_height = available_height / self.warehouse.height
        
        self.cell_size = min(cell_width, cell_height)
        self.cell_size = max(Config.MIN_CELL_SIZE, min(self.cell_size, Config.MAX_CELL_SIZE))
        
        # Calculate grid offset to center it
        grid_width = self.warehouse.width * self.cell_size
        grid_height = self.warehouse.height * self.cell_size
        
        self.offset_x = Config.SIDEBAR_WIDTH + (available_width - grid_width) / 2
        self.offset_y = Config.TOOLBAR_HEIGHT + (available_height - grid_height) / 2
    
    def draw(self, surface):
        """
        Draw the grid view.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw grid background
        grid_rect = pygame.Rect(
            self.offset_x, 
            self.offset_y, 
            self.warehouse.width * self.cell_size, 
            self.warehouse.height * self.cell_size
        )
        pygame.draw.rect(surface, Config.COLORS["background"], grid_rect)
        
        # Draw cells
        for row in range(self.warehouse.height):
            for col in range(self.warehouse.width):
                cell_rect = self._get_cell_rect(row, col)
                
                # Determine cell color
                color = Config.COLORS["empty"]
                
                # Check cell type
                if self.warehouse.is_wall(row, col):
                    color = Config.COLORS["wall"]
                elif (row, col) in self.path_cells:
                    color = Config.COLORS["path"]
                elif (row, col) in self.visited_cells:
                    color = Config.COLORS["visited"]
                elif (row, col) in self.frontier_cells:
                    color = Config.COLORS["frontier"]
                
                # Draw cell
                pygame.draw.rect(surface, color, cell_rect)
                
                # Draw cell border
                pygame.draw.rect(surface, Config.COLORS["grid_lines"], cell_rect, 1)
        
        # Draw start and end positions
        if self.warehouse.start_pos:
            start_rect = self._get_cell_rect(*self.warehouse.start_pos)
            pygame.draw.rect(surface, Config.COLORS["start"], start_rect)
            pygame.draw.rect(surface, Config.COLORS["grid_lines"], start_rect, 1)
            
            # Draw 'S' label
            font = pygame.font.SysFont(None, 24)
            text = font.render("S", True, Config.COLORS["button_text"])
            text_rect = text.get_rect(center=start_rect.center)
            surface.blit(text, text_rect)
        
        if self.warehouse.end_pos:
            end_rect = self._get_cell_rect(*self.warehouse.end_pos)
            pygame.draw.rect(surface, Config.COLORS["end"], end_rect)
            pygame.draw.rect(surface, Config.COLORS["grid_lines"], end_rect, 1)
            
            # Draw 'E' label
            font = pygame.font.SysFont(None, 24)
            text = font.render("E", True, Config.COLORS["button_text"])
            text_rect = text.get_rect(center=end_rect.center)
            surface.blit(text, text_rect)
        
        # Draw grid border
        pygame.draw.rect(surface, Config.COLORS["grid_lines"], grid_rect, 2)
    
    def get_cell_at_position(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get the grid cell at the given screen position.
        
        Args:
            pos: Screen position (x, y)
            
        Returns:
            Grid cell as (row, col) or None if out of bounds
        """
        x, y = pos
        
        # Check if position is inside the grid
        if (x < self.offset_x or 
            y < self.offset_y or 
            x >= self.offset_x + self.warehouse.width * self.cell_size or 
            y >= self.offset_y + self.warehouse.height * self.cell_size):
            return None
        
        # Calculate grid cell
        col = int((x - self.offset_x) / self.cell_size)
        row = int((y - self.offset_y) / self.cell_size)
        
        # Validate cell
        if 0 <= row < self.warehouse.height and 0 <= col < self.warehouse.width:
            return (row, col)
        
        return None
    
    def add_visited_cell(self, cell: Tuple[int, int]):
        """
        Add a cell to the visited cells set for visualization.
        
        Args:
            cell: Grid cell (row, col)
        """
        if cell != self.warehouse.start_pos and cell != self.warehouse.end_pos:
            self.visited_cells.add(cell)
    
    def add_frontier_cell(self, cell: Tuple[int, int]):
        """
        Add a cell to the frontier cells set for visualization.
        
        Args:
            cell: Grid cell (row, col)
        """
        if cell != self.warehouse.start_pos and cell != self.warehouse.end_pos:
            self.frontier_cells.add(cell)
    
    def add_path_cell(self, cell: Tuple[int, int]):
        """
        Add a cell to the path cells set for visualization.
        
        Args:
            cell: Grid cell (row, col)
        """
        if cell != self.warehouse.start_pos and cell != self.warehouse.end_pos:
            self.path_cells.add(cell)
    
    def clear_path_visualization(self):
        """Clear all path visualization sets."""
        self.visited_cells.clear()
        self.frontier_cells.clear()
        self.path_cells.clear()
    
    def _get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """
        Get the rectangle for a grid cell.
        
        Args:
            row: Grid row
            col: Grid column
            
        Returns:
            Pygame rectangle for the cell
        """
        return pygame.Rect(
            self.offset_x + col * self.cell_size,
            self.offset_y + row * self.cell_size,
            self.cell_size,
            self.cell_size
        )

class Toolbar:
    """Toolbar UI component for the warehouse optimizer."""
    
    def __init__(self, screen):
        """
        Initialize the toolbar.
        
        Args:
            screen: The parent screen
        """
        self.screen = screen
        self.height = Config.TOOLBAR_HEIGHT
        self.rect = pygame.Rect(
            Config.SIDEBAR_WIDTH, 
            0, 
            Config.WINDOW_WIDTH - Config.SIDEBAR_WIDTH, 
            self.height
        )
        
        # Initialize UI elements
        self._init_ui_elements()
    
    def _init_ui_elements(self):
        """Initialize the UI elements in the toolbar."""
        self.ui_elements = []
        padding = 10
        button_height = 36
        button_width = 120
        current_x = Config.SIDEBAR_WIDTH + padding
        
        # Save button
        save_button = Button(
            "Save",
            (current_x, (self.height - button_height) // 2),
            (button_width, button_height),
            self._on_save
        )
        self.ui_elements.append(save_button)
        current_x += button_width + 10
        
        # Load button
        load_button = Button(
            "Load",
            (current_x, (self.height - button_height) // 2),
            (button_width, button_height),
            self._on_load
        )
        self.ui_elements.append(load_button)
        current_x += button_width + 10
        
        # Clear all button
        clear_button = Button(
            "Clear All",
            (current_x, (self.height - button_height) // 2),
            (button_width, button_height),
            self._on_clear_all
        )
        self.ui_elements.append(clear_button)
        current_x += button_width + 20
        
        # Grid size controls
        size_label = Label(
            "Grid Size:",
            (current_x, self.height // 2),
            Config.FONT_SIZES["small"],
            Config.COLORS["text"],
            "left"
        )
        self.ui_elements.append(size_label)
        current_x += 70
        
        # Decrease width button
        decrease_width_button = Button(
            "-",
            (current_x, (self.height - button_height) // 2),
            (30, button_height),
            lambda: self._on_resize_grid(width_delta=-1)
        )
        self.ui_elements.append(decrease_width_button)
        current_x += 35
        
        # Width label
        self.width_label = Label(
            str(self.screen.warehouse.width),
            (current_x, self.height // 2),
            Config.FONT_SIZES["small"],
            Config.COLORS["text"],
            "center"
        )
        self.ui_elements.append(self.width_label)
        current_x += 25
        
        # Increase width button
        increase_width_button = Button(
            "+",
            (current_x, (self.height - button_height) // 2),
            (30, button_height),
            lambda: self._on_resize_grid(width_delta=1)
        )
        self.ui_elements.append(increase_width_button)
        current_x += 50
        
        # Decrease height button
        decrease_height_button = Button(
            "-",
            (current_x, (self.height - button_height) // 2),
            (30, button_height),
            lambda: self._on_resize_grid(height_delta=-1)
        )
        self.ui_elements.append(decrease_height_button)
        current_x += 35
        
        # Height label
        self.height_label = Label(
            str(self.screen.warehouse.height),
            (current_x, self.height // 2),
            Config.FONT_SIZES["small"],
            Config.COLORS["text"],
            "center"
        )
        self.ui_elements.append(self.height_label)
        current_x += 25
        
        # Increase height button
        increase_height_button = Button(
            "+",
            (current_x, (self.height - button_height) // 2),
            (30, button_height),
            lambda: self._on_resize_grid(height_delta=1)
        )
        self.ui_elements.append(increase_height_button)
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            True if the event was handled, False otherwise
        """
        # Check if event is inside toolbar
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
            if not self.rect.collidepoint(event.pos):
                return False
        
        # Pass event to UI elements
        for element in self.ui_elements:
            if hasattr(element, "handle_event") and element.handle_event(event):
                return True
        
        return False
    
    def update(self, dt):
        """
        Update the toolbar.
        
        Args:
            dt: Delta time in seconds
        """
        # Update UI elements
        for element in self.ui_elements:
            if hasattr(element, "update"):
                element.update(dt)
        
        # Update the rect
        self.rect.width = Config.WINDOW_WIDTH - Config.SIDEBAR_WIDTH
        
        # Update grid size labels
        self.width_label.set_text(str(self.screen.warehouse.width))
        self.height_label.set_text(str(self.screen.warehouse.height))
    
    def draw(self, surface):
        """
        Draw the toolbar.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw toolbar background
        pygame.draw.rect(surface, Config.COLORS["toolbar"], self.rect)
        
        # Draw UI elements
        for element in self.ui_elements:
            if hasattr(element, "draw"):
                element.draw(surface)
        
        # Draw toolbar border
        pygame.draw.line(
            surface, 
            Config.COLORS["grid_lines"], 
            (Config.SIDEBAR_WIDTH, self.height), 
            (Config.WINDOW_WIDTH, self.height), 
            2
        )
    
    def _on_save(self):
        """Handle save button click."""
        # This would normally open a save dialog
        # For simplicity, we'll just save to a fixed file
        self.screen.warehouse.save("warehouse.json")
    
    def _on_load(self):
        """Handle load button click."""
        # This would normally open a load dialog
        # For simplicity, we'll just try to load from a fixed file
        try:
            self.screen.warehouse = self.screen.warehouse.load("warehouse.json")
            self.screen.pathfinder.warehouse = self.screen.warehouse
            self.screen.clear_path()
        except Exception as e:
            print(f"Error loading warehouse: {e}")
    
    def _on_clear_all(self):
        """Handle clear all button click."""
        self.screen.warehouse.clear()
        self.screen.clear_path()
    
    def _on_resize_grid(self, width_delta=0, height_delta=0):
        """
        Handle grid resize.
        
        Args:
            width_delta: Change in width
            height_delta: Change in height
        """
        new_width = max(
            Config.MIN_GRID_SIZE, 
            min(
                self.screen.warehouse.width + width_delta, 
                Config.MAX_GRID_SIZE
            )
        )
        
        new_height = max(
            Config.MIN_GRID_SIZE, 
            min(
                self.screen.warehouse.height + height_delta, 
                Config.MAX_GRID_SIZE
            )
        )
        
        if (new_width != self.screen.warehouse.width or 
            new_height != self.screen.warehouse.height):
            self.screen.warehouse.resize(new_width, new_height)
            self.screen.clear_path()

class Sidebar:
    """Sidebar UI component for the warehouse optimizer."""
    
    def __init__(self, screen):
        """
        Initialize the sidebar.
        
        Args:
            screen: The parent screen
        """
        self.screen = screen
        self.width = Config.SIDEBAR_WIDTH
        self.rect = pygame.Rect(0, 0, self.width, Config.WINDOW_HEIGHT)
        
        # Initialize UI elements
        self._init_ui_elements()
    
    def _init_ui_elements(self):
        """Initialize the UI elements in the sidebar."""
        self.ui_elements = []
        padding = 20
        element_height = 40
        current_y = 20
        
        # Title
        title_label = Label(
            "Warehouse Optimizer",
            (self.width // 2, current_y),
            Config.FONT_SIZES["title"],
            Config.COLORS["text"],
            "center"
        )
        self.ui_elements.append(title_label)
        current_y += 60
        
        # Tool selection section
        tools_label = Label(
            "Tools",
            (padding, current_y),
            Config.FONT_SIZES["medium"],
            Config.COLORS["text"],
            "left"
        )
        self.ui_elements.append(tools_label)
        current_y += 30
        
        # Tool buttons
        button_width = (self.width - padding * 2 - 10) // 2
        
        # Wall button
        wall_button = Button(
            "Wall (1)",
            (padding, current_y),
            (button_width, element_height),
            self._on_wall_tool,
            is_selected=True
        )
        self.ui_elements.append(wall_button)
        self.wall_button = wall_button
        
        # Start button
        start_button = Button(
            "Start (2)",
            (padding + button_width + 10, current_y),
            (button_width, element_height),
            self._on_start_tool
        )
        self.ui_elements.append(start_button)
        self.start_button = start_button
        current_y += element_height + 10
        
        # End button
        end_button = Button(
            "End (3)",
            (padding, current_y),
            (button_width, element_height),
            self._on_end_tool
        )
        self.ui_elements.append(end_button)
        self.end_button = end_button
        
        # Erase button
        erase_button = Button(
            "Erase (4)",
            (padding + button_width + 10, current_y),
            (button_width, element_height),
            self._on_erase_tool
        )
        self.ui_elements.append(erase_button)
        self.erase_button = erase_button
        current_y += element_height + 30
        
        # A* settings section
        astar_label = Label(
            "A* Algorithm Settings",
            (padding, current_y),
            Config.FONT_SIZES["medium"],
            Config.COLORS["text"],
            "left"
        )
        self.ui_elements.append(astar_label)
        current_y += 30
        
        # Heuristic selection
        heuristic_label = Label(
            "Heuristic:",
            (padding, current_y),
            Config.FONT_SIZES["small"],
            Config.COLORS["text"],
            "left"
        )
        self.ui_elements.append(heuristic_label)
        current_y += 25
        
        # Heuristic buttons
        button_width = (self.width - padding * 2 - 20) // 3
        
        # Manhattan button
        manhattan_button = Button(
            "Manhattan",
            (padding, current_y),
            (button_width, element_height),
            lambda: self._on_heuristic("manhattan"),
            is_selected=True
        )
        self.ui_elements.append(manhattan_button)
        self.manhattan_button = manhattan_button
        
        # Euclidean button
        euclidean_button = Button(
            "Euclidean",
            (padding + button_width + 10, current_y),
            (button_width, element_height),
            lambda: self._on_heuristic("euclidean")
        )
        self.ui_elements.append(euclidean_button)
        self.euclidean_button = euclidean_button
        
        # Diagonal button
        diagonal_button = Button(
            "Diagonal",
            (padding + 2 * (button_width + 10), current_y),
            (button_width, element_height),
            lambda: self._on_heuristic("diagonal")
        )
        self.ui_elements.append(diagonal_button)
        self.diagonal_button = diagonal_button
        current_y += element_height + 30
        
        # Actions section
        actions_label = Label(
            "Actions",
            (padding, current_y),
            Config.FONT_SIZES["medium"],
            Config.COLORS["text"],
            "left"
        )
        self.ui_elements.append(actions_label)
        current_y += 30
        
        # Find path button
        find_path_button = Button(
            "Find Path (Space)",
            (padding, current_y),
            (self.width - padding * 2, element_height),
            self._on_find_path,
            color=Config.COLORS["success"]
        )
        self.ui_elements.append(find_path_button)
        current_y += element_height + 10
        
        # Clear path button
        clear_path_button = Button(
            "Clear Path (C)",
            (padding, current_y),
            (self.width - padding * 2, element_height),
            self._on_clear_path
        )
        self.ui_elements.append(clear_path_button)
        current_y += element_height + 10
        
        # Random grid button
        random_grid_button = Button(
            "Random Grid (R)",
            (padding, current_y),
            (self.width - padding * 2, element_height),
            self._on_random_grid
        )
        self.ui_elements.append(random_grid_button)
        current_y += element_height + 30
        
        # Metrics section
        metrics_label = Label(
            "Metrics",
            (padding, current_y),
            Config.FONT_SIZES["medium"],
            Config.COLORS["text"],
            "left"
        )
        self.ui_elements.append(metrics_label)
        current_y += 30
        
        # Metrics labels
        self.path_status_label = Label(
            "Path: Not found",
            (padding, current_y),
            Config.FONT_SIZES["small"],
            Config.COLORS["text_secondary"],
            "left"
        )
        self.ui_elements.append(self.path_status_label)
        current_y += 25
        
        self.path_length_label = Label(
            "Length: 0",
            (padding, current_y),
            Config.FONT_SIZES["small"],
            Config.COLORS["text_secondary"],
            "left"
        )
        self.ui_elements.append(self.path_length_label)
        current_y += 25
        
        self.nodes_expanded_label = Label(
            "Nodes expanded: 0",
            (padding, current_y),
            Config.FONT_SIZES["small"],
            Config.COLORS["text_secondary"],
            "left"
        )
        self.ui_elements.append(self.nodes_expanded_label)
        current_y += 25
        
        self.search_time_label = Label(
            "Search time: 0.000 ms",
            (padding, current_y),
            Config.FONT_SIZES["small"],
            Config.COLORS["text_secondary"],
            "left"
        )
        self.ui_elements.append(self.search_time_label)
        current_y += 25
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            True if the event was handled, False otherwise
        """
        # Check if event is inside sidebar
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
            if not self.rect.collidepoint(event.pos):
                return False
        
        # Pass event to UI elements
        for element in self.ui_elements:
            if hasattr(element, "handle_event") and element.handle_event(event):
                return True
        
        return False
    
    def update(self, dt):
        """
        Update the sidebar.
        
        Args:
            dt: Delta time in seconds
        """
        # Update UI elements
        for element in self.ui_elements:
            if hasattr(element, "update"):
                element.update(dt)
        
        # Update the rect
        self.rect.height = Config.WINDOW_HEIGHT
        
        # Update metrics
        metrics = self.screen.metrics
        
        self.path_status_label.set_text(
            f"Path: {'Found' if metrics['path_found'] else 'Not found'}"
        )
        self.path_length_label.set_text(
            f"Length: {metrics['path_length']}"
        )
        self.nodes_expanded_label.set_text(
            f"Nodes expanded: {metrics['nodes_expanded']}"
        )
        self.search_time_label.set_text(
            f"Search time: {metrics['search_time']*1000:.2f} ms"
        )
        
        # Update tool button states
        self.wall_button.is_selected = self.screen.current_tool == "wall"
        self.start_button.is_selected = self.screen.current_tool == "start"
        self.end_button.is_selected = self.screen.current_tool == "end"
        self.erase_button.is_selected = self.screen.current_tool == "erase"
        
        # Update heuristic button states
        current_heuristic = self.screen.pathfinder.current_heuristic
        self.manhattan_button.is_selected = current_heuristic == "manhattan"
        self.euclidean_button.is_selected = current_heuristic == "euclidean"
        self.diagonal_button.is_selected = current_heuristic == "diagonal"
    
    def draw(self, surface):
        """
        Draw the sidebar.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw sidebar background
        pygame.draw.rect(surface, Config.COLORS["sidebar"], self.rect)
        
        # Draw UI elements
        for element in self.ui_elements:
            if hasattr(element, "draw"):
                element.draw(surface)
        
        # Draw sidebar border
        pygame.draw.line(
            surface, 
            Config.COLORS["grid_lines"], 
            (self.width, 0), 
            (self.width, Config.WINDOW_HEIGHT), 
            2
        )
    
    def _on_wall_tool(self):
        """Handle wall tool button click."""
        self.screen.set_tool("wall")
    
    def _on_start_tool(self):
        """Handle start tool button click."""
        self.screen.set_tool("start")
    
    def _on_end_tool(self):
        """Handle end tool button click."""
        self.screen.set_tool("end")
    
    def _on_erase_tool(self):
        """Handle erase tool button click."""
        self.screen.set_tool("erase")
    
    def _on_heuristic(self, heuristic):
        """
        Handle heuristic button click.
        
        Args:
            heuristic: Heuristic name
        """
        self.screen.set_heuristic(heuristic)
    
    def _on_find_path(self):
        """Handle find path button click."""
        self.screen.find_path()
    
    def _on_clear_path(self):
        """Handle clear path button click."""
        self.screen.clear_path()
    
    def _on_random_grid(self):
        """Handle random grid button click."""
        self.screen.generate_random_warehouse(0.2)

class WarehouseScreen:
    """Main screen for the warehouse optimizer application."""
    
    def __init__(self, app):
        """
        Initialize the warehouse screen.
        
        Args:
            app: The main application instance
        """
        self.app = app
        
        # Create a new warehouse model
        self.warehouse = Warehouse(Config.DEFAULT_GRID_WIDTH, Config.DEFAULT_GRID_HEIGHT)
        
        # Initialize A* pathfinder
        self.pathfinder = AStarPathfinder(self.warehouse)
        
        # UI components
        self.sidebar = Sidebar(self)
        self.toolbar = Toolbar(self)
        self.grid_view = GridView(self.warehouse, self.pathfinder)
        
        # Animation state
        self.animating = False
        self.animation_step = 0
        self.animation_timer = 0
        self.animation_path = []
        self.animation_visited = []
        self.animation_frontier = []
        
        # Tool state
        self.current_tool = "wall"  # wall, start, end, erase
        
        # Mouse state
        self.mouse_down = False
        self.last_cell = None
        
        # Performance metrics
        self.metrics = {
            "path_found": False,
            "path_length": 0,
            "nodes_expanded": 0,
            "search_time": 0
        }
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
        """
        # Let UI components handle events first
        if self.sidebar.handle_event(event):
            return
        if self.toolbar.handle_event(event):
            return
        
        # Handle mouse events for grid interaction
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                self.mouse_down = True
                self._handle_mouse_interaction(event.pos)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.mouse_down = False
                self.last_cell = None
        
        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_down:
                self._handle_mouse_interaction(event.pos)
        
        # Handle keyboard events
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.find_path()
            elif event.key == pygame.K_c:
                self.clear_path()
            elif event.key == pygame.K_r:
                self.warehouse.generate_random(0.2)
                self.clear_path()
            elif event.key == pygame.K_1:
                self.set_tool("wall")
            elif event.key == pygame.K_2:
                self.set_tool("start")
            elif event.key == pygame.K_3:
                self.set_tool("end")
            elif event.key == pygame.K_4:
                self.set_tool("erase")
    
    def update(self, dt):
        """
        Update the screen logic.
        
        Args:
            dt: Delta time since last update in seconds
        """
        # Update UI components
        self.sidebar.update(dt)
        self.toolbar.update(dt)
        self.grid_view.update(dt)
        
        # Update path animation
        if self.animating:
            self.animation_timer += dt
            if self.animation_timer >= Config.ANIMATION_SPEED:
                self.animation_timer = 0
                self._advance_animation()
    
    def draw(self, surface):
        """
        Draw the screen.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw the grid
        self.grid_view.draw(surface)
        
        # Draw UI components
        self.toolbar.draw(surface)
        self.sidebar.draw(surface)
    
    def find_path(self):
        """Find and animate the optimal path using A* algorithm."""
        if not self.warehouse.start_pos or not self.warehouse.end_pos:
            return
        
        # Clear any existing path
        self.clear_path()
        
        # Start the animation
        self.animating = True
        self.animation_step = 0
        self.animation_timer = 0
        
        # Get the complete path
        path = self.pathfinder.find_path(
            self.warehouse.start_pos, 
            self.warehouse.end_pos
        )
        
        # Store the animation data
        self.animation_path = self.pathfinder.path.copy()
        self.animation_visited = self.pathfinder.visited_cells.copy()
        self.animation_frontier = []
        
        # Update metrics
        self.metrics = self.pathfinder.get_metrics()
    
    def clear_path(self):
        """Clear the current path and reset animation state."""
        self.animating = False
        self.animation_step = 0
        self.animation_timer = 0
        self.animation_path = []
        self.animation_visited = []
        self.animation_frontier = []
        self.grid_view.clear_path_visualization()
        
        # Reset metrics
        self.metrics = {
            "path_found": False,
            "path_length": 0,
            "nodes_expanded": 0,
            "search_time": 0
        }
    
    def set_tool(self, tool):
        """
        Set the current editing tool.
        
        Args:
            tool: Tool name ('wall', 'start', 'end', 'erase')
        """
        self.current_tool = tool
    
    def set_heuristic(self, heuristic):
        """
        Set the A* heuristic function.
        
        Args:
            heuristic: Heuristic name ('manhattan', 'euclidean', 'diagonal')
        """
        self.pathfinder.set_heuristic(heuristic)
    
    def generate_random_warehouse(self, wall_density=0.2):
        """
        Generate a random warehouse layout.
        
        Args:
            wall_density: Probability of a cell being a wall (0.0 to 1.0)
        """
        self.warehouse.generate_random(wall_density)
        self.clear_path()
    
    def _handle_mouse_interaction(self, mouse_pos):
        """
        Handle mouse interaction with the grid.
        
        Args:
            mouse_pos: Mouse position (x, y)
        """
        # Convert mouse position to grid cell
        cell = self.grid_view.get_cell_at_position(mouse_pos)
        if not cell:
            return
        
        row, col = cell
        
        # Skip if same cell as last interaction to prevent repeated actions
        if cell == self.last_cell:
            return
        
        self.last_cell = cell
        
        # Apply the current tool
        if self.current_tool == "wall":
            self.warehouse.set_wall(row, col, True)
            self.clear_path()
        elif self.current_tool == "erase":
            self.warehouse.set_wall(row, col, False)
            self.clear_path()
        elif self.current_tool == "start":
            self.warehouse.set_start(row, col)
            self.clear_path()
        elif self.current_tool == "end":
            self.warehouse.set_end(row, col)
            self.clear_path()
    
    def _advance_animation(self):
        """Advance the path finding animation by one step."""
        if not self.animation_visited and not self.animation_path:
            self.animating = False
            return
        
        # First show the visited cells
        if self.animation_visited:
            step_size = max(1, len(self.animation_visited) // 30)  # Show multiple cells per step for faster animation
            for _ in range(min(step_size, len(self.animation_visited))):
                if self.animation_visited:
                    cell = self.animation_visited.pop(0)
                    self.grid_view.add_visited_cell(cell)
        
        # Then show the final path
        elif self.animation_path:
            if self.animation_step < len(self.animation_path):
                self.grid_view.add_path_cell(self.animation_path[self.animation_step])
                self.animation_step += 1
            else:
                self.animating = False

class WarehouseOptimizer:
    """Main application class for the Warehouse Optimizer."""
    
    def __init__(self):
        """Initialize the Warehouse Optimizer application."""
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Warehouse Optimizer - A* Algorithm")
        
        # Set up the display
        self.screen = pygame.display.set_mode(
            (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), 
            pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        
        # Set up font
        pygame.font.init()
        
        # Load assets
        self.load_assets()
        
        # Create the warehouse screen
        self.current_screen = WarehouseScreen(self)
        
        # Running flag
        self.running = True
    
    def load_assets(self):
        """Load all necessary assets for the application."""
        # Create asset directories if they don't exist
        os.makedirs("assets", exist_ok=True)
    
    def run(self):
        """Run the main application loop."""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Update window size
                    Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT = event.size
                    self.screen = pygame.display.set_mode(
                        (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), 
                        pygame.RESIZABLE
                    )
                
                # Pass events to current screen
                self.current_screen.handle_event(event)
            
            # Update current screen
            self.current_screen.update(self.clock.get_time() / 1000.0)
            
            # Draw current screen
            self.screen.fill(Config.COLORS["background"])
            self.current_screen.draw(self.screen)
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(Config.FPS)
        
        # Clean up
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = WarehouseOptimizer()
    app.run()