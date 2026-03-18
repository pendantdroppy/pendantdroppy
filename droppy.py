#!/usr/bin/env python3
from __future__ import annotations
import sys
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("This script requires opencv-python. Install with: pip install opencv-python") from e

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
        QFileDialog, QMessageBox, QTabWidget, QGroupBox, QFormLayout,
        QProgressBar, QTextEdit, QDialog, QScrollArea, QCheckBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage
except ImportError:
    raise SystemExit("PyQt6 is required. Install with: pip install PyQt6")

global user

try:
    user = os.environ['USER']
except error:
    user =  os.getenv('USERNAME')

@dataclass
class Inputs:
    image_path: str
    droplet_type: str
    needle_left_xy: Tuple[float, float]
    needle_right_xy: Tuple[float, float]
    needle_diameter_mm: float
    sigma: float
    canny1: int
    canny2: int
    num_points: int
    direction: int
    min_r_mm: float
    min_z_mm: float
    circle_window: int
    deriv_window: int
    stable_s_min_frac: float
    stable_s_max_frac: float
    rmse_factor: float
    mad_z: float
    circle_window_frac: float


# ============================================================
# Utilities
# ============================================================
def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float64)

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n

def line_signed_value(x: float, y: float, p1: np.ndarray, p2: np.ndarray) -> float:
    return (x - p1[0]) * (p2[1] - p1[1]) - (y - p1[1]) * (p2[0] - p1[0])

def line_signed_grid(h: int, w: int, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return (X - x1) * (y2 - y1) - (Y - y1) * (x2 - x1)

def gaussian_blur(img_gray: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img_gray, (k, k), sigmaX=sigma, sigmaY=sigma)

def droplet_halfplane_mask(
    img_shape: Tuple[int, int],
    needle_left: Tuple[float, float],
    needle_right: Tuple[float, float],
    droplet_type: str,
) -> np.ndarray:
    h, w = img_shape
    p1 = np.array(needle_left, dtype=np.float64)
    p2 = np.array(needle_right, dtype=np.float64)
    f = line_signed_grid(h, w, p1, p2)
    midpoint = 0.5 * (p1 + p2)
    test_point = (midpoint[0], midpoint[1] - 10.0)
    fk_top = line_signed_value(test_point[0], test_point[1], p1, p2)
    if fk_top == 0:
        fk_top = 1.0
    top_is_positive = fk_top > 0
    if droplet_type == "rising":
        mask = (f > 0) if top_is_positive else (f < 0)
    elif droplet_type == "pendant":
        mask = (f < 0) if top_is_positive else (f > 0)
    else:
        raise ValueError("droplet_type must be 'rising' or 'pendant'")
    return (mask.astype(np.uint8) * 255)

def find_edges_masked(img_gray: np.ndarray, sigma: float, canny1: int, canny2: int, mask: np.ndarray) -> np.ndarray:
    blurred = gaussian_blur(img_gray, sigma)
    masked = blurred.copy()
    masked[mask == 0] = 0
    edges = cv2.Canny(masked, canny1, canny2, L2gradient=True)
    edges[mask == 0] = 0
    return edges

def nearest_index(points_xy: np.ndarray, pt_xy: Tuple[float, float]) -> int:
    d2 = (points_xy[:, 0] - pt_xy[0]) ** 2 + (points_xy[:, 1] - pt_xy[1]) ** 2
    return int(np.argmin(d2))

def cyclic_slice(arr: np.ndarray, start: int, length: int, direction: int) -> np.ndarray:
    n = len(arr)
    idxs = [(start + direction * i) % n for i in range(length)]
    return arr[idxs]

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def fit_circle_kasa(points_xy: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
    if points_xy.shape[0] < 3:
        return np.array([np.nan, np.nan], dtype=np.float64), float("nan"), float("nan"), False
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = sol
        cx = -D / 2.0
        cy = -E / 2.0
        r2 = (D * D + E * E) / 4.0 - F
        if not np.isfinite(r2) or r2 <= 1e-9:
            return np.array([cx, cy], dtype=np.float64), float("nan"), float("nan"), False
        r = float(np.sqrt(r2))
        dx = x - cx
        dy = y - cy
        dist = np.sqrt(dx * dx + dy * dy)
        resid = dist - r
        rmse = float(np.sqrt(np.mean(resid * resid)))
        return np.array([cx, cy], dtype=np.float64), r, rmse, True
    except Exception:
        return np.array([np.nan, np.nan], dtype=np.float64), float("nan"), float("nan"), False

def local_linear_slope(x: np.ndarray, y: np.ndarray, half_window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        i0 = max(0, i - half_window)
        i1 = min(n, i + half_window + 1)
        xx = x[i0:i1]
        yy = y[i0:i1]
        ok = np.isfinite(xx) & np.isfinite(yy)
        if np.count_nonzero(ok) < 3:
            continue
        xx = xx[ok]
        yy = yy[ok]
        x0 = float(np.mean(xx))
        y0 = float(np.mean(yy))
        X = xx - x0
        Y = yy - y0
        denom = float(np.dot(X, X))
        if denom < 1e-18:
            continue
        out[i] = float(np.dot(X, Y) / denom)
    return out

def mad_filter(x: np.ndarray, z: float = 3.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-18:
        return np.isfinite(x)
    return np.isfinite(x) & (np.abs(x - med) <= z * mad)

def integrate_young_laplace(Bo: float, droplet_type: str, z_stop: float = 3.0, ds: float = 0.002):
    """Non-dimensional Young–Laplace integration for axisymmetric drop."""
    r = 1e-6
    z = 0.0
    phi = 0.0
    s = 0.0
    r_list = [r]
    z_list = [z]
    g_sign = -1.0 if droplet_type == "pendant" else +1.0
    while z < z_stop:
        def f(state):
            rr, zz, pp = state
            dr = np.cos(pp)
            dz = np.sin(pp)
            dphi = 2.0 - (g_sign * Bo) * zz - np.sin(pp) / max(rr, 1e-8)
            return np.array([dr, dz, dphi], dtype=np.float64)
        y = np.array([r, z, phi], dtype=np.float64)
        k1 = f(y)
        k2 = f(y + 0.5 * ds * k1)
        k3 = f(y + 0.5 * ds * k2)
        k4 = f(y + ds * k3)
        y = y + ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        r, z, phi = float(y[0]), float(y[1]), float(y[2])
        s += ds
        if s > 20:
            break
        if r < 0 or r > 10:
            break
        if abs(phi) > 4:
            break
        if not np.isfinite(r) or not np.isfinite(z) or not np.isfinite(phi):
            break
        if phi > np.pi:
            break
        r_list.append(r)
        z_list.append(z)
    return np.array(r_list, dtype=np.float64), np.array(z_list, dtype=np.float64)


def check_yl_needle_interception(
    r_star_pred: np.ndarray,
    z_star_pred: np.ndarray,
    R0_len_mm: float,
    tip: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    lensing_factor: float,
    mm_per_px: float,
) -> Tuple[bool, float]:
    """Check if YL curve intercepts the needle at the tip."""
    if len(r_star_pred) < 10:
        return False, float('nan')
    
    # Convert YL curve to pixels
    r_mm_pred = r_star_pred * R0_len_mm
    z_mm_pred = z_star_pred * R0_len_mm
    r_px_pred = (r_mm_pred / mm_per_px) / lensing_factor
    z_px_pred = z_mm_pred / mm_per_px
    
    # Needle edge is at r_px = 0 (the axis at z=0)
    # We're looking for where the YL curve crosses r_px ≈ 0
    # This happens at the lowest z values in the curve
    
    # Get the portion of the curve near z ≈ 0 to R0
    z_threshold = R0_len_mm / mm_per_px * 1.5  # Within ~1.5× R0 in pixels
    near_tip_mask = (z_px_pred >= -1.0) & (z_px_pred <= z_threshold)
    
    if not np.any(near_tip_mask):
        return False, float('nan')
    
    r_near_tip = r_px_pred[near_tip_mask]
    
    # Minimum r value tells us how close we get to the axis
    min_r = np.min(np.abs(r_near_tip))
    
    # If min_r is very small (< 1 pixel), we're intercepting
    intercepts = min_r < 2.0  # Within ~2 pixels of needle axis
    
    return intercepts, min_r


def auto_correct_parameters_for_interception(
    current_sigma: float,
    current_canny1: int,
    current_plateau_length: int,
    current_Bo: float,
    r_star_pred: np.ndarray,
    z_star_pred: np.ndarray,
    closest_distance_px: float,
    step: int = 1,
) -> Tuple[float, int, int]:
    """
    Auto-adjust parameters to improve YL curve interception with needle.
    For rising droplets, if the YL curve doesn't reach the needle, we need to:
    1. Increase sigma to smooth edges (better edge detection upstream)
    2. Adjust canny1 to catch weaker edges
    3. Possibly adjust plateau_length
    
    Args:
        step: Iteration number (1=first attempt, 2=more aggressive, etc.)
    
    Returns:
        (new_sigma, new_canny1, new_plateau_length)
    """
    new_sigma = current_sigma
    new_canny1 = current_canny1
    new_plateau_length = current_plateau_length
    
    # Distance from needle edge (larger = worse)
    if closest_distance_px > 3.0:
        # Curve is far from needle
        if step == 1:
            # First try: smooth more, lower canny threshold
            new_sigma = min(current_sigma * 1.3, 4.0)
            new_canny1 = max(current_canny1 - 10, 20)
        elif step == 2:
            # Second try: even more smoothing
            new_sigma = min(current_sigma * 1.6, 5.0)
            new_canny1 = max(current_canny1 - 20, 10)
            new_plateau_length = int(current_plateau_length * 0.85)  # Smaller plateau
        else:
            # Third try: aggressive adjustments
            new_sigma = min(current_sigma * 2.0, 6.0)
            new_canny1 = max(current_canny1 - 30, 5)
            new_plateau_length = int(current_plateau_length * 0.7)
    
    return new_sigma, new_canny1, new_plateau_length


class YLCurveGenetic:
    """Genetic algorithm for rising droplet YL curve optimization."""
    
    def __init__(
        self,
        initial_sigma: float,
        initial_canny1: int,
        initial_plateau_length: int,
        initial_low_clarity: float,
        initial_high_clarity: float,
        initial_bo: float = 0.2,
        pop_size: int = 8,
        max_generations: int = 30,
        mutation_rate: float = 0.15,
        convergence_threshold: float = 0.001,
    ):
        self.initial_sigma = initial_sigma
        self.initial_canny1 = initial_canny1
        self.initial_plateau_length = initial_plateau_length
        self.initial_low_clarity = initial_low_clarity
        self.initial_high_clarity = initial_high_clarity
        self.initial_bo = initial_bo
        
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        
        self.generation = 0
        self.best_fitness_history = []
        self.generation_images = []
        self.best_individual = None
        self.best_fitness = -float('inf')
    
    def create_individual(self, sigma=None, canny1=None, plateau_length=None, low_clarity=None, high_clarity=None, bo=None):
        """Create a parameter individual."""
        return {
            'sigma': sigma if sigma is not None else self.initial_sigma,
            'canny1': canny1 if canny1 is not None else self.initial_canny1,
            'plateau_length': plateau_length if plateau_length is not None else self.initial_plateau_length,
            'low_clarity': low_clarity if low_clarity is not None else self.initial_low_clarity,
            'high_clarity': high_clarity if high_clarity is not None else self.initial_high_clarity,
            'bo': bo if bo is not None else self.initial_bo,
        }
    
    def mutate_individual(self, individual):
        """Mutate individual with strong random parameter changes."""
        mutant = individual.copy()
        
        # ALWAYS mutate all six parameters (guarantee diversity)
        # Mutate sigma: ±50%
        factor = np.random.uniform(0.5, 1.5)
        mutant['sigma'] = np.clip(mutant['sigma'] * factor, 0.5, 5.0)
        
        # Mutate canny1: ±50
        delta = np.random.randint(-50, 51)
        mutant['canny1'] = np.clip(mutant['canny1'] + delta, 5, 150)
        
        # Mutate plateau_length: ±30%
        factor = np.random.uniform(0.7, 1.3)
        mutant['plateau_length'] = int(np.clip(
            mutant['plateau_length'] * factor, 50, 150
        ))
        
        # Mutate low_clarity: ±40%
        factor = np.random.uniform(0.6, 1.4)
        mutant['low_clarity'] = np.clip(mutant['low_clarity'] * factor, 0.05, 0.5)
        
        # Mutate high_clarity: ±40%
        factor = np.random.uniform(0.6, 1.4)
        mutant['high_clarity'] = np.clip(mutant['high_clarity'] * factor, 0.005, 0.15)
        
        # Mutate Bo: ±50%
        factor = np.random.uniform(0.5, 1.5)
        mutant['bo'] = np.clip(mutant['bo'] * factor, 0.01, 1.0)
        
        return mutant
    
    def crossover(self, parent1, parent2):
        """Create offspring from two parents."""
        # Blend parameters from both parents
        child = {
            'sigma': (parent1['sigma'] + parent2['sigma']) / 2.0 + np.random.normal(0, 0.1),
            'canny1': int((parent1['canny1'] + parent2['canny1']) / 2.0 + np.random.normal(0, 2)),
            'plateau_length': int((parent1['plateau_length'] + parent2['plateau_length']) / 2.0),
            'low_clarity': (parent1['low_clarity'] + parent2['low_clarity']) / 2.0 + np.random.normal(0, 0.01),
            'high_clarity': (parent1['high_clarity'] + parent2['high_clarity']) / 2.0 + np.random.normal(0, 0.005),
            'bo': (parent1['bo'] + parent2['bo']) / 2.0 + np.random.normal(0, 0.02),
        }
        
        # Ensure bounds
        child['sigma'] = np.clip(child['sigma'], 0.5, 5.0)
        child['canny1'] = np.clip(child['canny1'], 5, 150)
        child['plateau_length'] = np.clip(child['plateau_length'], 50, 150)
        child['low_clarity'] = np.clip(child['low_clarity'], 0.05, 0.5)
        child['high_clarity'] = np.clip(child['high_clarity'], 0.005, 0.15)
        child['bo'] = np.clip(child['bo'], 0.01, 1.0)
        
        return child
    
    def calculate_fitness(
        self,
        displacement_px: float,
        rmse_mm: float,
        bo_error: float,
        plateau_fitness: float,
    ) -> float:
        """
        Calculate multi-objective fitness score.
        Higher is better.
        
        Args:
            displacement_px: Distance to needle corner (lower is better)
            rmse_mm: Circle detection RMSE in mm (lower is better)
            bo_error: YL fit error (lower is better)
            plateau_fitness: Fit quality on plateau (0-1, higher is better)
        
        Returns:
            Fitness score (higher is better)
        """
        # Normalize components (using inverse/negative for "minimize" objectives)
        displacement_score = 1.0 / (1.0 + displacement_px)  # 1 at d=0, decays
        rmse_score = 1.0 / (1.0 + rmse_mm * 100)  # Scaled for mm units
        error_score = 1.0 / (1.0 + bo_error)
        
        # Weighted combination (displacement is most critical)
        fitness = (
            0.5 * displacement_score +
            0.2 * rmse_score +
            0.15 * error_score +
            0.15 * plateau_fitness
        )
        
        return fitness
    
    def evolve(self, evaluate_func, output_dir: str):
        """
        Main evolutionary loop.
        
        Args:
            evaluate_func: Function that takes parameters dict and returns
                          (fitness, metrics_dict, image_array)
            output_dir: Directory to save generation images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize population with diversity
        population = []
        for i in range(self.pop_size):
            if i == 0:
                # First individual: use defaults
                population.append(self.create_individual())
            else:
                # Other individuals: add random variations
                sigma = self.initial_sigma * np.random.uniform(0.7, 1.3)
                canny1 = int(self.initial_canny1 + np.random.randint(-30, 31))
                plateau = int(self.initial_plateau_length * np.random.uniform(0.8, 1.2))
                low_clarity = self.initial_low_clarity * np.random.uniform(0.7, 1.3)
                high_clarity = self.initial_high_clarity * np.random.uniform(0.7, 1.3)
                bo = self.initial_bo * np.random.uniform(0.7, 1.3)
                
                sigma = np.clip(sigma, 0.5, 5.0)
                canny1 = np.clip(canny1, 5, 150)
                plateau = np.clip(plateau, 50, 150)
                low_clarity = np.clip(low_clarity, 0.05, 0.5)
                high_clarity = np.clip(high_clarity, 0.005, 0.15)
                bo = np.clip(bo, 0.01, 1.0)
                
                population.append(self.create_individual(sigma, canny1, plateau, low_clarity, high_clarity, bo))
        
        for gen in range(self.max_generations):
            self.generation = gen
            
            # Evaluate population
            fitnesses = []
            individuals_data = []
            
            
            for ind in population:
                fitness, metrics, gen_image = evaluate_func(ind)
                fitnesses.append(fitness)
                individuals_data.append((ind, fitness, metrics, gen_image))
            
            # Sort by fitness
            individuals_data.sort(key=lambda x: x[1], reverse=True)
            fitnesses.sort(reverse=True)
            
            # Track best
            if fitnesses[0] > self.best_fitness:
                self.best_fitness = fitnesses[0]
                self.best_individual = individuals_data[0][0].copy()
            
            self.best_fitness_history.append(self.best_fitness)
            
            # Save generation image
            if individuals_data[0][3] is not None:
                gen_img_path = os.path.join(output_dir, f"gen_{gen:03d}_best.png")
                cv2.imwrite(gen_img_path, individuals_data[0][3])
                self.generation_images.append(gen_img_path)
            
            # Check convergence - require at least 10 generations and meaningful improvement
            if len(self.best_fitness_history) >= 10:
                recent_improvement = (
                    self.best_fitness_history[-1] - self.best_fitness_history[-10]
                )
                # Need at least 0.01 improvement per 10 generations
                if abs(recent_improvement) < 0.01:
                    break
            
            # Selection + crossover + mutation
            # Keep top performers (elitism)
            elite_count = max(2, self.pop_size // 4)
            new_population = [ind for ind, _, _, _ in individuals_data[:elite_count]]
            
            # Fill rest with crossover + mutation
            while len(new_population) < self.pop_size:
                # Select parents (tournament selection)
                p1_idx = np.random.randint(0, elite_count)
                p2_idx = np.random.randint(0, elite_count)
                parent1 = individuals_data[p1_idx][0]
                parent2 = individuals_data[p2_idx][0]
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate_individual(child)
                
                new_population.append(child)
            
            population = new_population
        
        return self.best_individual, self.best_fitness_history




class YLCurveFallingDroplet:
    """Genetic algorithm for falling droplet YL curve optimization."""
    
    def __init__(
        self,
        initial_sigma: float,
        initial_canny1: int,
        initial_plateau_length: int,
        initial_low_clarity: float,
        initial_high_clarity: float,
        initial_bo: float = 0.2,
        pop_size: int = 8,
        max_generations: int = 30,
        mutation_rate: float = 0.15,
        convergence_threshold: float = 0.001,
    ):
        self.initial_sigma = initial_sigma
        self.initial_canny1 = initial_canny1
        self.initial_plateau_length = initial_plateau_length
        self.initial_low_clarity = initial_low_clarity
        self.initial_high_clarity = initial_high_clarity
        self.initial_bo = initial_bo
        
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        
        self.generation = 0
        self.best_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
    
    def create_individual(self, sigma=None, canny1=None, plateau_length=None, low_clarity=None, high_clarity=None, bo=None):
        """Create a parameter individual."""
        return {
            'sigma': sigma if sigma is not None else self.initial_sigma,
            'canny1': canny1 if canny1 is not None else self.initial_canny1,
            'plateau_length': plateau_length if plateau_length is not None else self.initial_plateau_length,
            'low_clarity': low_clarity if low_clarity is not None else self.initial_low_clarity,
            'high_clarity': high_clarity if high_clarity is not None else self.initial_high_clarity,
            'bo': bo if bo is not None else self.initial_bo,
        }
    
    def mutate_individual(self, individual):
        """Mutate an individual with random parameter changes."""
        mutant = individual.copy()
        
        # ALWAYS mutate all six parameters
        factor = np.random.uniform(0.5, 1.5)
        mutant['sigma'] = np.clip(mutant['sigma'] * factor, 0.5, 5.0)
        
        delta = np.random.randint(-50, 51)
        mutant['canny1'] = np.clip(mutant['canny1'] + delta, 5, 150)
        
        factor = np.random.uniform(0.7, 1.3)
        mutant['plateau_length'] = int(np.clip(
            mutant['plateau_length'] * factor, 50, 150
        ))
        
        factor = np.random.uniform(0.6, 1.4)
        mutant['low_clarity'] = np.clip(mutant['low_clarity'] * factor, 0.05, 0.5)
        
        factor = np.random.uniform(0.6, 1.4)
        mutant['high_clarity'] = np.clip(mutant['high_clarity'] * factor, 0.005, 0.15)
        
        factor = np.random.uniform(0.5, 1.5)
        mutant['bo'] = np.clip(mutant['bo'] * factor, 0.01, 1.0)
        
        return mutant
    
    def crossover(self, parent1, parent2):
        """Create offspring from two parents."""
        child = {
            'sigma': (parent1['sigma'] + parent2['sigma']) / 2.0 + np.random.normal(0, 0.1),
            'canny1': int((parent1['canny1'] + parent2['canny1']) / 2.0 + np.random.normal(0, 2)),
            'plateau_length': int((parent1['plateau_length'] + parent2['plateau_length']) / 2.0),
            'low_clarity': (parent1['low_clarity'] + parent2['low_clarity']) / 2.0 + np.random.normal(0, 0.01),
            'high_clarity': (parent1['high_clarity'] + parent2['high_clarity']) / 2.0 + np.random.normal(0, 0.005),
            'bo': (parent1['bo'] + parent2['bo']) / 2.0 + np.random.normal(0, 0.02),
        }
        
        child['sigma'] = np.clip(child['sigma'], 0.5, 5.0)
        child['canny1'] = np.clip(child['canny1'], 5, 150)
        child['plateau_length'] = np.clip(child['plateau_length'], 50, 150)
        child['low_clarity'] = np.clip(child['low_clarity'], 0.05, 0.5)
        child['high_clarity'] = np.clip(child['high_clarity'], 0.005, 0.15)
        child['bo'] = np.clip(child['bo'], 0.01, 1.0)
        
        return child
    
    def evolve(self, evaluate_func, output_dir: str):
        """Main evolutionary loop for falling droplet."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize population with diversity
        population = []
        for i in range(self.pop_size):
            if i == 0:
                population.append(self.create_individual())
            else:
                sigma = self.initial_sigma * np.random.uniform(0.7, 1.3)
                canny1 = int(self.initial_canny1 + np.random.randint(-30, 31))
                plateau = int(self.initial_plateau_length * np.random.uniform(0.8, 1.2))
                low_clarity = self.initial_low_clarity * np.random.uniform(0.7, 1.3)
                high_clarity = self.initial_high_clarity * np.random.uniform(0.7, 1.3)
                bo = self.initial_bo * np.random.uniform(0.7, 1.3)
                
                sigma = np.clip(sigma, 0.5, 5.0)
                canny1 = np.clip(canny1, 5, 150)
                plateau = np.clip(plateau, 50, 150)
                low_clarity = np.clip(low_clarity, 0.05, 0.5)
                high_clarity = np.clip(high_clarity, 0.005, 0.15)
                bo = np.clip(bo, 0.01, 1.0)
                
                population.append(self.create_individual(sigma, canny1, plateau, low_clarity, high_clarity, bo))
        
        for gen in range(self.max_generations):
            self.generation = gen
            
            individuals_data = []
            fitnesses = []
            
            for ind in population:
                fitness, metrics, gen_image = evaluate_func(ind)
                fitnesses.append(fitness)
                individuals_data.append((ind, fitness, metrics, gen_image))
            
            individuals_data.sort(key=lambda x: x[1], reverse=True)
            fitnesses.sort(reverse=True)
            
            if fitnesses[0] > self.best_fitness:
                self.best_fitness = fitnesses[0]
                self.best_individual = individuals_data[0][0].copy()
            
            self.best_fitness_history.append(self.best_fitness)
            
            # Check convergence
            if len(self.best_fitness_history) >= 10:
                recent_improvement = (
                    self.best_fitness_history[-1] - self.best_fitness_history[-10]
                )
                if abs(recent_improvement) < 0.01:
                    break
            
            # Selection + crossover + mutation
            elite_count = max(2, self.pop_size // 4)
            new_population = [ind for ind, _, _, _ in individuals_data[:elite_count]]
            
            while len(new_population) < self.pop_size:
                p1 = individuals_data[np.random.randint(0, len(individuals_data) // 2)][0]
                p2 = individuals_data[np.random.randint(0, len(individuals_data) // 2)][0]
                child = self.crossover(p1, p2)
                child = self.mutate_individual(child)
                new_population.append(child)
            
            population = new_population
        
        return self.best_individual, self.best_fitness_history


class ProcessingThread(QThread):
    """Worker thread that runs the exact original pipeline."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, inputs: Inputs, ui_params: dict):
        super().__init__()
        self.inputs = inputs
        self.ui_params = ui_params
    
    def run(self):
        try:
            self.progress.emit("Loading image...")
            img = cv2.imread(self.inputs.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {self.inputs.image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H_img, W_img = gray.shape[:2]
            
            # Calibration from needle
            self.progress.emit("Computing px/mm calibration...")
            nl = np.array(self.inputs.needle_left_xy, dtype=np.float64)
            nr = np.array(self.inputs.needle_right_xy, dtype=np.float64)
            needle_len_px = float(np.linalg.norm(nr - nl))
            if needle_len_px < 1e-6:
                raise ValueError("Needle endpoints too close")
            
            px_per_mm = needle_len_px / self.inputs.needle_diameter_mm
            mm_per_px = self.inputs.needle_diameter_mm / needle_len_px
            
            # Mask
            self.progress.emit("Creating mask...")
            droplet_mask = droplet_halfplane_mask(
                img_shape=(H_img, W_img),
                needle_left=self.inputs.needle_left_xy,
                needle_right=self.inputs.needle_right_xy,
                droplet_type=self.inputs.droplet_type,
            )
            
            # Edges
            self.progress.emit("Finding edges...")
            edges = find_edges_masked(gray, self.inputs.sigma, self.inputs.canny1, self.inputs.canny2, droplet_mask)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                raise ValueError("No contours found")
            
            contours_xy = [c.reshape(-1, 2).astype(np.float64) for c in contours]
            
            # Filter by ROI
            self.progress.emit("Filtering contours...")
            x0, y0, rw, rh = self.ui_params["roi_box"]
            x1 = x0 + rw
            y1 = y0 + rh
            filtered_contours = []
            for c in contours_xy:
                mask_roi = (
                    (c[:, 0] >= x0) & (c[:, 0] <= x1) &
                    (c[:, 1] >= y0) & (c[:, 1] <= y1)
                )
                c_roi = c[mask_roi]
                if len(c_roi) > 50:
                    filtered_contours.append(c_roi)
            
            if not filtered_contours:
                raise ValueError("No contour points in ROI")
            
            contour_xy_full = max(filtered_contours, key=lambda c: len(c))
            
            # Extract extreme points
            self.progress.emit("Extracting droplet geometry...")
            contour = contour_xy_full
            tol = 2.0
            
            y_min = np.min(contour[:, 1])
            top_band = contour[np.abs(contour[:, 1] - y_min) <= tol]
            top_x = float(np.mean(top_band[:, 0]))
            top_point = np.array([top_x, float(y_min)], dtype=np.float64)
            
            y_max = np.max(contour[:, 1])
            bottom_band = contour[np.abs(contour[:, 1] - y_max) <= tol]
            bottom_x = float(np.mean(bottom_band[:, 0]))
            bottom_point = np.array([bottom_x, float(y_max)], dtype=np.float64)
            
            if self.inputs.droplet_type == "rising":
                extreme_xy = (float(top_point[0]), float(top_point[1]))
            else:
                extreme_xy = (float(bottom_point[0]), float(bottom_point[1]))
            
            axis_x = extreme_xy[0]
            
            tol_lr = 2.0
            x_min = np.min(contour[:, 0])
            left_band = contour[np.abs(contour[:, 0] - x_min) <= tol_lr]
            left_x = float(np.mean(left_band[:, 0]))
            left_y = float(np.mean(left_band[:, 1]))
            left_extreme = np.array([left_x, left_y], dtype=np.float64)
            
            x_max = np.max(contour[:, 0])
            right_band = contour[np.abs(contour[:, 0] - x_max) <= tol_lr]
            right_x = float(np.mean(right_band[:, 0]))
            right_y = float(np.mean(right_band[:, 1]))
            right_extreme = np.array([right_x, right_y], dtype=np.float64)
            
            width_mid_y = 0.5 * (left_extreme[1] + right_extreme[1])
            center_xy = (float(axis_x), float(width_mid_y))
            
            # Sample contour
            start_idx = nearest_index(contour_xy_full, extreme_xy)
            sample_len = min(self.inputs.num_points, len(contour_xy_full))
            pts_xy = cyclic_slice(contour_xy_full, start_idx, sample_len, self.inputs.direction)
            M = len(pts_xy)
            
            # Coordinate system
            self.progress.emit("Setting up coordinate system...")
            tip = np.array(extreme_xy, dtype=np.float64)
            ctr = np.array(center_xy, dtype=np.float64)
            lft = np.array(left_extreme, dtype=np.float64)
            rht = np.array(right_extreme, dtype=np.float64)
            
            R0 = ctr - tip
            ey = unit(R0)
            ex = rot90(ey)
            R0_len_px = float(np.linalg.norm(R0))
            R0_len_mm = R0_len_px * mm_per_px
            
            # Arc length
            diffs = np.diff(pts_xy, axis=0)
            ds_px = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
            s_px = np.zeros(M, dtype=np.float64)
            if M > 1:
                s_px[1:] = np.cumsum(ds_px)
            s_mm = s_px * mm_per_px
            s_total = float(s_mm[-1]) if M > 1 else 0.0
            
            # Coordinates - LENSING FACTOR APPLIED HERE
            deltas_tip = pts_xy - tip
            lensing_factor = float(self.ui_params["lensing_factor"])
            r_p_px = (deltas_tip @ ex) * lensing_factor  # Apply lensing to measured r
            z_p_px = deltas_tip @ ey
            r_p_mm = r_p_px * mm_per_px
            z_p_mm = z_p_px * mm_per_px
            
            side = np.ones_like(r_p_px)
            side[r_p_px < 0] = -1.0
            ex_ref = side[:, None] * ex[None, :]
            
            r_star = r_p_mm / R0_len_mm
            z_star = z_p_mm / R0_len_mm
            
            # Define run_pipeline function - EXACTLY like original
            def run_pipeline(W_val, lensing_factor_val):
                """Run single pipeline with given circle_window."""
                circle_c = np.full((M, 2), np.nan, dtype=np.float64)
                circle_r_mm = np.full(M, np.nan, dtype=np.float64)
                circle_rmse_px = np.full(M, np.nan, dtype=np.float64)
                circle_ok = np.zeros(M, dtype=bool)
                t_unit = np.full((M, 2), np.nan, dtype=np.float64)
                
                for i in range(M):
                    i0 = max(0, i - W_val)
                    i1 = min(M, i + W_val + 1)
                    window_pts = pts_xy[i0:i1]
                    cxy, rad_px, rmse_px, ok = fit_circle_kasa(window_pts)
                    circle_c[i] = cxy
                    circle_r_mm[i] = rad_px * mm_per_px if np.isfinite(rad_px) else np.nan
                    circle_rmse_px[i] = rmse_px
                    circle_ok[i] = ok
                    
                    if not ok:
                        continue
                    rvec = pts_xy[i] - cxy
                    rn = np.linalg.norm(rvec)
                    if rn < 1e-9:
                        circle_ok[i] = False
                        continue
                    rhat = rvec / rn
                    that = rot90(rhat)
                    tn = np.linalg.norm(that)
                    if tn < 1e-12:
                        circle_ok[i] = False
                        continue
                    t_unit[i] = that / tn
                
                # Phi mapping
                phi = np.full(M, np.nan, dtype=np.float64)
                dot_tx = np.full(M, np.nan, dtype=np.float64)
                ok_t = circle_ok & np.isfinite(t_unit[:, 0]) & np.isfinite(t_unit[:, 1])
                dot_tx[ok_t] = np.sum(t_unit[ok_t] * ex_ref[ok_t], axis=1)
                alpha = np.full(M, np.nan, dtype=np.float64)
                alpha[ok_t] = np.arccos(clamp01(np.abs(dot_tx[ok_t])))
                between = (z_p_mm >= -1e-9) & (z_p_mm <= R0_len_mm + 1e-9)
                phi[ok_t & between] = alpha[ok_t & between]
                phi[ok_t & (~between)] = np.pi - alpha[ok_t & (~between)]
                
                if np.isfinite(phi[0]):
                    phi = phi - phi[0]
                    phi = np.clip(phi, 0.0, np.pi)
                
                dphi_ds_local = local_linear_slope(s_mm, phi, half_window=int(self.inputs.deriv_window))
                dphi_ds_star = R0_len_mm * dphi_ds_local
                sinphi = np.sin(phi)
                
                valid_basic = (
                    np.isfinite(phi) &
                    np.isfinite(dphi_ds_star) &
                    ok_t &
                    (np.abs(r_star) >= self.inputs.min_r_mm / R0_len_mm) &
                    (np.abs(z_star) >= self.inputs.min_z_mm / R0_len_mm)
                )
                
                Bo_i = np.full(M, np.nan, dtype=np.float64)
                Bo_i[valid_basic] = (
                    2.0
                    - dphi_ds_star[valid_basic]
                    - (sinphi[valid_basic] / r_star[valid_basic])
                ) / z_star[valid_basic]
                
                ok_rmse = circle_ok & np.isfinite(circle_rmse_px)
                rmse_med = float(np.median(circle_rmse_px[ok_rmse])) if np.any(ok_rmse) else float("nan")
                rmse_thresh = self.inputs.rmse_factor * rmse_med if np.isfinite(rmse_med) else float("nan")
                
                # Convert RMSE from pixels to mm
                rmse_med_mm = rmse_med * mm_per_px if np.isfinite(rmse_med) else float("nan")
                rmse_thresh_mm = rmse_thresh * mm_per_px if np.isfinite(rmse_thresh) else float("nan")
                
                if s_total > 1e-9:
                    s_frac = s_mm / s_total
                else:
                    s_frac = np.zeros_like(s_mm)
                
                stable_gate = (
                    valid_basic &
                    np.isfinite(Bo_i) &
                    np.isfinite(circle_rmse_px) &
                    (circle_rmse_px <= rmse_thresh) &
                    (s_frac >= self.inputs.stable_s_min_frac) &
                    (s_frac <= self.inputs.stable_s_max_frac)
                )
                
                # PLATEAU ANALYSIS - CENTER SYMMETRICALLY ON EQUATOR
                plateau_len = int(self.ui_params["plateau_width"])
                if plateau_len < 3:
                    return None
                
                equator_z = R0_len_mm
                equator_idx = int(np.argmin(np.abs(z_p_mm - equator_z)))
                half_len = plateau_len // 2
                
                # Center symmetrically around equator_idx
                plateau_start_i = equator_idx - half_len
                plateau_end_i = equator_idx + half_len
                
                # Handle boundaries - shift if necessary but try to keep equator in middle
                if plateau_start_i < 0:
                    plateau_start_i = 0
                    plateau_end_i = min(M - 1, plateau_len - 1)
                if plateau_end_i >= M:
                    plateau_end_i = M - 1
                    plateau_start_i = max(0, M - plateau_len)
                
                plateau_mask = np.zeros(M, dtype=bool)
                plateau_mask[plateau_start_i:plateau_end_i + 1] = True
                
                Bo_plateau_raw = Bo_i[plateau_mask & np.isfinite(Bo_i)]
                if Bo_plateau_raw.size < 10:
                    return None
                
                r_meas = r_star[plateau_mask]
                z_meas = z_star[plateau_mask]
                ok_m = np.isfinite(r_meas) & np.isfinite(z_meas)
                r_meas = r_meas[ok_m]
                z_meas = z_meas[ok_m]
                if r_meas.size < 10:
                    return None
                
                order = np.argsort(z_meas)
                z_meas_s = z_meas[order]
                r_meas_s = r_meas[order]
                
                bo_min_plateau = float(np.nanmin(Bo_plateau_raw))
                bo_max_plateau = float(np.nanmax(Bo_plateau_raw))
                bo_wiggle = float(self.ui_params.get("bo_wiggle_room", 0.0))
                bo_min_eff = max(1e-6, bo_min_plateau - bo_wiggle)
                bo_max_eff = bo_max_plateau + bo_wiggle
                
                if not np.isfinite(bo_min_eff) or not np.isfinite(bo_max_eff) or bo_max_eff <= bo_min_eff:
                    return None
                
                Bo_candidates = np.linspace(bo_min_eff, bo_max_eff, 50)
                best_Bo = None
                best_error = float("inf")
                
                for Bo in Bo_candidates:
                    r_pred, z_pred = integrate_young_laplace(float(Bo), self.inputs.droplet_type, z_stop=3.0)
                    if len(z_pred) < 10:
                        continue
                    
                    okp = np.isfinite(z_pred) & np.isfinite(r_pred)
                    z_pred = z_pred[okp]
                    r_pred = r_pred[okp]
                    
                    if z_pred.size < 10:
                        continue
                    
                    if not np.all(np.diff(z_pred) >= -1e-12):
                        o2 = np.argsort(z_pred)
                        z_pred = z_pred[o2]
                        r_pred = r_pred[o2]
                    
                    z0 = float(z_pred[0])
                    z1 = float(z_pred[-1])
                    zq = np.clip(z_meas_s, z0, z1)
                    
                    r_interp = np.interp(zq, z_pred, r_pred)
                    rms = float(np.sqrt(np.mean((r_meas_s - r_interp) ** 2)))
                    overall_error = rms / np.sqrt(max(1, plateau_len))
                    
                    if np.isfinite(overall_error) and overall_error < best_error:
                        best_error = overall_error
                        best_Bo = float(Bo)
                
                if best_Bo is None:
                    return None
                
                Bo_plateau = Bo_plateau_raw.copy()
                if Bo_plateau.size >= 8:
                    keep = mad_filter(Bo_plateau, z=self.inputs.mad_z)
                    Bo_plateau = Bo_plateau[keep]
                
                return {
                    "best_Bo": best_Bo,
                    "best_error": best_error,
                    "plateau_mask": plateau_mask,
                    "circle_c": circle_c,
                    "circle_r_mm": circle_r_mm,
                    "circle_rmse_px": circle_rmse_px,
                    "t_unit": t_unit,
                    "dot_tx": dot_tx,
                    "phi": phi,
                    "dphi_ds_local": dphi_ds_local,
                    "Bo_i": Bo_i,
                    "stable_gate": stable_gate,
                    "rmse_med": rmse_med,
                    "rmse_thresh": rmse_thresh,
                    "rmse_med_mm": rmse_med_mm,
                    "rmse_thresh_mm": rmse_thresh_mm,
                    "Bo_plateau_raw": Bo_plateau_raw,
                    "Bo_plateau_filtered": Bo_plateau,
                    "plateau_start_i": plateau_start_i,
                    "plateau_end_i": plateau_end_i,
                    "plateau_length": int(np.count_nonzero(plateau_mask)),
                }
            
            self.progress.emit("Running low clarity analysis...")
            W_low = max(1, int(round(float(self.ui_params["low_clarity_ratio"]) * M)))
            result_low = run_pipeline(W_low, lensing_factor)
            
            self.progress.emit("Running high clarity analysis...")
            W_high = max(1, int(round(float(self.ui_params["high_clarity_ratio"]) * M)))
            result_high = run_pipeline(W_high, lensing_factor)
            
            if result_low is None or result_high is None:
                raise ValueError("Pipeline failed for one or both clarity runs")
            
            # GENETIC ALGORITHM OPTIMIZATION (Rising droplets only)
            use_ga = self.ui_params.get("use_genetic_algorithm", False)
            ga_best_params = None
            ga_enabled_final = False
            
            if use_ga and self.inputs.droplet_type == "rising":
                self.progress.emit("Running genetic algorithm optimization...")
                
                # Create evaluation function that the GA will call
                def evaluate_params(params_dict):
                    """
                    Evaluate a parameter set by running full pipeline with new parameters.
                    Returns: (fitness, metrics_dict, visualization_image)
                    """
                    # Extract parameters
                    sigma_test = float(params_dict['sigma'])
                    canny1_test = int(params_dict['canny1'])
                    plateau_len_test = int(params_dict['plateau_length'])
                    low_clarity_test = float(params_dict.get('low_clarity', self.ui_params.get("low_clarity_ratio", 0.1)))
                    high_clarity_test = float(params_dict.get('high_clarity', self.ui_params.get("high_clarity_ratio", 0.01)))
                    
                    # Re-run edge detection with new sigma and canny1
                    edges_test = find_edges_masked(gray, sigma_test, canny1_test, self.inputs.canny2, droplet_mask)
                    
                    contours_test, _ = cv2.findContours(edges_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if not contours_test:
                        # Error: lower sigma and retry once
                        sigma_reduced = max(0.5, sigma_test * 0.8)
                        edges_test = find_edges_masked(gray, sigma_reduced, canny1_test, self.inputs.canny2, droplet_mask)
                        contours_test, _ = cv2.findContours(edges_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if not contours_test:
                            return 0.001, {"error": "No contours (even after sigma reduction)"}, None
                        sigma_test = sigma_reduced
                    
                    contours_xy_test = [c.reshape(-1, 2).astype(np.float64) for c in contours_test]
                    
                    # Filter by ROI
                    x0_roi, y0_roi, rw_roi, rh_roi = self.ui_params["roi_box"]
                    x1_roi = x0_roi + rw_roi
                    y1_roi = y0_roi + rh_roi
                    filtered = []
                    for c in contours_xy_test:
                        mask = (c[:, 0] >= x0_roi) & (c[:, 0] <= x1_roi) & (c[:, 1] >= y0_roi) & (c[:, 1] <= y1_roi)
                        c_filt = c[mask]
                        if len(c_filt) > 50:
                            filtered.append(c_filt)
                    
                    if not filtered:
                        return 0.001, {"error": "No ROI contours"}, None
                    
                    contour_test = max(filtered, key=lambda c: len(c))
                    
                    # Use sampled contour points
                    start_idx_test = nearest_index(contour_test, extreme_xy)
                    pts_test = cyclic_slice(contour_test, start_idx_test, sample_len, self.inputs.direction)
                    M_test = len(pts_test)
                    
                    if M_test < 20:
                        return 0.001, {"error": "Too few points"}, None
                    
                    # Calculate circle window with HIGH clarity ratio
                    W_test_high = max(1, int(round(high_clarity_test * M_test)))
                    circle_c_test = np.full((M_test, 2), np.nan, dtype=np.float64)
                    circle_rmse_px_test = np.full(M_test, np.nan, dtype=np.float64)
                    
                    for i in range(M_test):
                        i0 = max(0, i - W_test_high)
                        i1 = min(M_test, i + W_test_high + 1)
                        window_pts = pts_test[i0:i1]
                        cxy, rad_px, rmse_px, ok = fit_circle_kasa(window_pts)
                        circle_c_test[i] = cxy
                        circle_rmse_px_test[i] = rmse_px
                    
                    ok_rmse_test = np.isfinite(circle_rmse_px_test)
                    if not np.any(ok_rmse_test):
                        return 0.001, {"error": "No valid RMSE"}, None
                    
                    rmse_med_test = float(np.median(circle_rmse_px_test[ok_rmse_test]))
                    rmse_mm_test = rmse_med_test * mm_per_px
                    
                    # Convert to droplet coordinates
                    deltas_test = pts_test - tip
                    r_p_px_test = (deltas_test @ ex)
                    z_p_px_test = deltas_test @ ey
                    r_p_mm_test = r_p_px_test * mm_per_px
                    z_p_mm_test = z_p_px_test * mm_per_px
                    r_star_test = r_p_mm_test / R0_len_mm
                    z_star_test = z_p_mm_test / R0_len_mm
                    
                    # Calculate displacement (primary objective)
                    near_tip_mask = (z_p_px_test >= -2.0) & (z_p_px_test <= R0_len_mm / mm_per_px * 2.0)
                    if not np.any(near_tip_mask):
                        displacement = 999
                    else:
                        r_near = np.abs(r_p_px_test[near_tip_mask])
                        displacement = float(np.min(r_near))
                    
                    # Calculate plateau-based Bo fitness
                    # Use plateau_len_test to select plateau region
                    equator_z_test = R0_len_mm
                    equator_idx_test = int(np.argmin(np.abs(z_p_mm_test - equator_z_test)))
                    half_len_test = plateau_len_test // 2
                    
                    plateau_start_i_test = equator_idx_test - half_len_test
                    plateau_end_i_test = equator_idx_test + half_len_test
                    
                    # Boundary handling
                    if plateau_start_i_test < 0:
                        plateau_start_i_test = 0
                        plateau_end_i_test = min(M_test - 1, plateau_len_test - 1)
                    if plateau_end_i_test >= M_test:
                        plateau_end_i_test = M_test - 1
                        plateau_start_i_test = max(0, M_test - plateau_len_test)
                    
                    plateau_mask_test = np.zeros(M_test, dtype=bool)
                    plateau_mask_test[plateau_start_i_test:plateau_end_i_test + 1] = True
                    
                    r_meas_test = r_star_test[plateau_mask_test]
                    z_meas_test = z_star_test[plateau_mask_test]
                    ok_m_test = np.isfinite(r_meas_test) & np.isfinite(z_meas_test)
                    r_meas_test = r_meas_test[ok_m_test]
                    z_meas_test = z_meas_test[ok_m_test]
                    
                    if r_meas_test.size < 10:
                        return 0.001, {"error": "Too few plateau points"}, None
                    
                    # USE THE BO FROM THE INDIVIDUAL (not estimated)
                    bo_test = float(params_dict.get('bo', 0.2))
                    
                    # Generate YL curve with this Bo
                    try:
                        r_yl, z_yl = integrate_young_laplace(bo_test, "rising", z_stop=3.0)
                        if len(z_yl) < 10:
                            return 0.001, {"error": "YL integration failed"}, None
                    except:
                        return 0.001, {"error": "YL error"}, None
                    
                    # Calculate fit error to plateau
                    z0 = float(z_yl[0])
                    z1 = float(z_yl[-1])
                    order = np.argsort(z_meas_test)
                    z_s = z_meas_test[order]
                    r_s = r_meas_test[order]
                    
                    zq = np.clip(z_s, z0, z1)
                    r_interp = np.interp(zq, z_yl, r_yl)
                    bo_error = float(np.sqrt(np.mean((r_s - r_interp) ** 2)))
                    
                    # Calculate distance from YL curve to the needle corner point
                    # BUT offset the target point UP by ~8px to better follow the droplet curve
                    # This accounts for the natural curvature of the droplet surface
                    needle_corner_xy = np.array(self.inputs.needle_left_xy, dtype=np.float64)
                    # Offset UP (toward tip, negative ey direction) by 8 pixels
                    needle_intercept_target = needle_corner_xy - 8.0 * ey  # 8px UP from corner
                    
                    needle_target_rel_tip = needle_intercept_target - tip  # Vector from tip to target
                    
                    # Project onto droplet coordinate system
                    needle_target_r = needle_target_rel_tip @ ex  # Radial component (in pixels)
                    needle_target_z = needle_target_rel_tip @ ey  # Axial component (in pixels)
                    
                    # Convert to normalized droplet coordinates
                    needle_target_r_norm = (needle_target_r * mm_per_px) / R0_len_mm
                    needle_target_z_norm = (needle_target_z * mm_per_px) / R0_len_mm
                    
                    # Now find the closest point on the YL curve to this offset target point
                    # YL curve is in (r, z) normalized coordinates
                    distances_to_target = np.sqrt((r_yl - needle_target_r_norm)**2 + (z_yl - needle_target_z_norm)**2)
                    min_distance_to_needle = float(np.min(distances_to_target))
                    
                    # Convert back to pixels for reporting
                    needle_intercept_distance = min_distance_to_needle * R0_len_mm / mm_per_px
                    
                    # Penalty if it doesn't intercept (distance should be small)
                    intercept_score = 1.0 / (1.0 + needle_intercept_distance)
                    
                    # Plateau fitness: How well YL curve fits the measured plateau points
                    # bo_error is the RMSE between YL and ALL measured radii on plateau
                    plateau_fitness = 1.0 / (1.0 + bo_error)  # Higher fitness when RMSE is lower
                    
                    # SIMPLE CONSTRAINT: YL curve cannot be closer to centerline (smaller radius)
                    # than the needle intercept point UNTIL it is below that point
                    # BUT: Only applies BELOW the equator (z > 0)
                    closeness_penalty = 0.0
                    
                    # Find YL points that are ABOVE the intercept point (z < needle_target_z_norm)
                    # AND below the equator (z > 0)
                    # These should NOT be closer to centerline than the intercept
                    above_intercept_mask = (z_yl < needle_target_z_norm) & (z_yl > 0)
                    
                    if np.any(above_intercept_mask):
                        above_r = r_yl[above_intercept_mask]
                        # Any point above intercept that's narrower than intercept gets penalized
                        violations = above_r < needle_target_r_norm
                        if np.any(violations):
                            # Sum up the violations
                            violation_amounts = needle_target_r_norm - above_r[violations]
                            violation_px = violation_amounts * R0_len_mm / mm_per_px
                            closeness_penalty = 0.1 * np.sum(violation_px)
                    
                    # ADDITIONAL: Score how well YL curve passes through the lowest point of the plateau
                    # Find the point in the plateau with the lowest z-value (deepest into droplet)
                    if len(z_s) > 0:
                        lowest_z_idx = np.argmin(z_s)  # Index of lowest z in sorted plateau
                        lowest_z = z_s[lowest_z_idx]
                        lowest_r_measured = r_s[lowest_z_idx]
                        
                        # Interpolate YL curve at this z-value
                        if z0 <= lowest_z <= z1:
                            lowest_r_yl = float(np.interp(lowest_z, z_yl, r_yl))
                            # Calculate error at lowest point
                            lowest_point_error = abs(lowest_r_measured - lowest_r_yl)
                            # Score: how close YL is to this lowest plateau point
                            lowest_point_score = 1.0 / (1.0 + lowest_point_error)
                        else:
                            lowest_point_score = 0.0  # YL curve doesn't span to lowest point
                    else:
                        lowest_point_score = 0.0
                    
                    # Calculate fitness with clear goal:
                    # PRIMARY: YL curve must intercept the END of the plateau (lowest point)
                    # SECONDARY: YL curve should also reach the needle corner
                    # TERTIARY: Overall fit quality to all points
                    # CONSTRAINT: Below equator only - no narrowing above intercept point
                    
                    fitness = (
                        0.4 * lowest_point_score +                 # END of plateau fit (PRIMARY - 40%)
                        0.2 * intercept_score +                    # Needle corner reach (SECONDARY - 20%)
                        0.25 * plateau_fitness +                   # Overall plateau fit (25%)
                        0.15 / (1.0 + rmse_mm_test * 100) -        # Circle quality (15%)
                        closeness_penalty                          # Penalty if curve narrower than intercept above it (below equator only)
                    )
                    
                    metrics = {
                        "sigma": f"{sigma_test:.2f}",
                        "canny1": canny1_test,
                        "plateau": plateau_len_test,
                        "Bo": f"{bo_test:.4f}",
                        "RMSE_mm": f"{rmse_mm_test:.6f}",
                        "intercept_px": f"{needle_intercept_distance:.2f}",
                        "lowest_point_score": f"{lowest_point_score:.4f}",
                        "bo_error": f"{bo_error:.6f}",
                    }
                    
                    return fitness, metrics, None
                
                # Run GA
                outdir_ga = self.ui_params["roi_box"]  # Hack: use roi_box for now, should be output dir
                outdir_ga = self.ui_params.get("output_dir", "droppy_out")
                
                ga = YLCurveGenetic(
                    initial_sigma=self.inputs.sigma,
                    initial_canny1=self.inputs.canny1,
                    initial_plateau_length=self.ui_params["plateau_width"],
                    initial_low_clarity=self.ui_params.get("low_clarity_ratio", 0.1),
                    initial_high_clarity=self.ui_params.get("high_clarity_ratio", 0.01),
                    initial_bo=0.2,  # Default starting Bo (will be optimized)
                    pop_size=self.ui_params.get("ga_population", 8),
                    max_generations=self.ui_params.get("ga_generations", 30),
                )
                
                ga_best_params, ga_fitness_history = ga.evolve(evaluate_params, outdir_ga + "/ga_generations")
                ga_enabled_final = True
            
            elif use_ga and self.inputs.droplet_type == "pendant":
                self.progress.emit("Running genetic algorithm optimization (falling droplet)...")
                
                # Create evaluation function for falling droplet
                def evaluate_params_falling(params_dict):
                    """
                    Evaluate a parameter set for FALLING droplet.
                    Goal: YL curve intercepts both equator point (right) and first plateau point
                    Returns: (fitness, metrics_dict, visualization_image)
                    """
                    sigma_test = float(params_dict['sigma'])
                    canny1_test = int(params_dict['canny1'])
                    plateau_len_test = int(params_dict['plateau_length'])
                    low_clarity_test = float(params_dict.get('low_clarity', self.ui_params.get("low_clarity_ratio", 0.1)))
                    high_clarity_test = float(params_dict.get('high_clarity', self.ui_params.get("high_clarity_ratio", 0.01)))
                    bo_test = float(params_dict.get('bo', 0.2))
                    
                    # Re-run edge detection with new sigma and canny1
                    edges_test = find_edges_masked(gray, sigma_test, canny1_test, self.inputs.canny2, droplet_mask)
                    
                    contours_test, _ = cv2.findContours(edges_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if not contours_test:
                        # Error: lower sigma and retry once
                        sigma_reduced = max(0.5, sigma_test * 0.8)
                        edges_test = find_edges_masked(gray, sigma_reduced, canny1_test, self.inputs.canny2, droplet_mask)
                        contours_test, _ = cv2.findContours(edges_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if not contours_test:
                            return 0.001, {"error": "No contours (even after sigma reduction)"}, None
                        sigma_test = sigma_reduced
                    
                    contours_xy_test = [c.reshape(-1, 2).astype(np.float64) for c in contours_test]
                    
                    # Filter by ROI
                    x0_roi, y0_roi, rw_roi, rh_roi = self.ui_params["roi_box"]
                    x1_roi = x0_roi + rw_roi
                    y1_roi = y0_roi + rh_roi
                    filtered = []
                    for c in contours_xy_test:
                        mask = (c[:, 0] >= x0_roi) & (c[:, 0] <= x1_roi) & (c[:, 1] >= y0_roi) & (c[:, 1] <= y1_roi)
                        c_filt = c[mask]
                        if len(c_filt) > 50:
                            filtered.append(c_filt)
                    
                    if not filtered:
                        return 0.001, {"error": "No ROI contours"}, None
                    
                    contour_test = max(filtered, key=lambda c: len(c))
                    
                    # Use sampled contour points
                    start_idx_test = nearest_index(contour_test, extreme_xy)
                    pts_test = cyclic_slice(contour_test, start_idx_test, sample_len, self.inputs.direction)
                    M_test = len(pts_test)
                    
                    if M_test < 20:
                        return 0.001, {"error": "Too few points"}, None
                    
                    # Circle fitting
                    W_test_high = max(1, int(round(high_clarity_test * M_test)))
                    circle_c_test = np.full((M_test, 2), np.nan, dtype=np.float64)
                    circle_rmse_px_test = np.full(M_test, np.nan, dtype=np.float64)
                    
                    for i in range(M_test):
                        i0 = max(0, i - W_test_high)
                        i1 = min(M_test, i + W_test_high + 1)
                        window_pts = pts_test[i0:i1]
                        cxy, rad_px, rmse_px, ok = fit_circle_kasa(window_pts)
                        circle_c_test[i] = cxy
                        circle_rmse_px_test[i] = rmse_px
                    
                    ok_rmse_test = np.isfinite(circle_rmse_px_test)
                    if not np.any(ok_rmse_test):
                        return 0.001, {"error": "No valid RMSE"}, None
                    
                    rmse_med_test = float(np.median(circle_rmse_px_test[ok_rmse_test]))
                    rmse_mm_test = rmse_med_test * mm_per_px
                    
                    # Convert to droplet coordinates
                    deltas_test = pts_test - tip
                    r_p_px_test = (deltas_test @ ex)
                    z_p_px_test = deltas_test @ ey
                    r_p_mm_test = r_p_px_test * mm_per_px
                    z_p_mm_test = z_p_px_test * mm_per_px
                    r_star_test = r_p_mm_test / R0_len_mm
                    z_star_test = z_p_mm_test / R0_len_mm
                    
                    # For FALLING droplet: find equator point (z=0) and first plateau point
                    equator_idx_test = int(np.argmin(np.abs(z_p_mm_test)))
                    
                    # Find plateau region (for falling: above equator in droplet coords, which is negative z)
                    equator_z_test = -R0_len_mm  # For pendant, equator is negative z
                    plateau_start_i_test = int(np.argmin(np.abs(z_p_mm_test - equator_z_test)))
                    half_len_test = plateau_len_test // 2
                    plateau_start_i_test = max(0, plateau_start_i_test - half_len_test)
                    plateau_end_i_test = min(M_test - 1, plateau_start_i_test + plateau_len_test)
                    
                    # Get plateau measurements
                    plateau_mask_test = np.zeros(M_test, dtype=bool)
                    plateau_mask_test[plateau_start_i_test:plateau_end_i_test + 1] = True
                    
                    r_meas_test = r_star_test[plateau_mask_test]
                    z_meas_test = z_star_test[plateau_mask_test]
                    ok_m_test = np.isfinite(r_meas_test) & np.isfinite(z_meas_test)
                    r_meas_test = r_meas_test[ok_m_test]
                    z_meas_test = z_meas_test[ok_m_test]
                    
                    if r_meas_test.size < 5:
                        return 0.001, {"error": "Too few plateau points"}, None
                    
                    # Generate YL curve with this Bo
                    try:
                        r_yl, z_yl = integrate_young_laplace(bo_test, "pendant", z_stop=3.0)
                        if len(z_yl) < 10:
                            return 0.001, {"error": "YL integration failed"}, None
                    except:
                        return 0.001, {"error": "YL error"}, None
                    
                    # Calculate fit error to plateau
                    z0 = float(z_yl[0])
                    z1 = float(z_yl[-1])
                    order = np.argsort(z_meas_test)
                    z_s = z_meas_test[order]
                    r_s = r_meas_test[order]
                    
                    zq = np.clip(z_s, z0, z1)
                    r_interp = np.interp(zq, z_yl, r_yl)
                    bo_error = float(np.sqrt(np.mean((r_s - r_interp) ** 2)))
                    
                    # SCORE 1: Distance to equator point (right side, z=0)
                    equator_r_measured = float(r_star_test[equator_idx_test])
                    # Find YL at z=0
                    equator_yl_idx = int(np.argmin(np.abs(z_yl)))
                    equator_r_yl = float(r_yl[equator_yl_idx])
                    equator_error = abs(equator_r_measured - equator_r_yl)
                    equator_score = 1.0 / (1.0 + equator_error)
                    
                    # SCORE 2: Distance to first plateau point
                    if len(z_s) > 0:
                        first_plateau_idx = 0
                        first_plateau_z = z_s[first_plateau_idx]
                        first_plateau_r = r_s[first_plateau_idx]
                        
                        if z0 <= first_plateau_z <= z1:
                            first_plateau_r_yl = float(np.interp(first_plateau_z, z_yl, r_yl))
                            first_plateau_error = abs(first_plateau_r - first_plateau_r_yl)
                            first_plateau_score = 1.0 / (1.0 + first_plateau_error)
                        else:
                            first_plateau_score = 0.0
                    else:
                        first_plateau_score = 0.0
                    
                    # Overall plateau fit
                    plateau_fitness = 1.0 / (1.0 + bo_error)
                    
                    # Fitness for falling droplet
                    fitness = (
                        0.4 * equator_score +              # Equator point (40%)
                        0.3 * first_plateau_score +        # First plateau point (30%)
                        0.2 * plateau_fitness +            # Overall fit (20%)
                        0.1 / (1.0 + rmse_mm_test * 100)  # Circle quality (10%)
                    )
                    
                    metrics = {
                        "sigma": f"{sigma_test:.2f}",
                        "canny1": canny1_test,
                        "plateau": plateau_len_test,
                        "Bo": f"{bo_test:.4f}",
                        "equator_score": f"{equator_score:.4f}",
                        "first_plateau_score": f"{first_plateau_score:.4f}",
                    }
                    
                    return fitness, metrics, None
                
                # Run falling droplet GA
                outdir_ga = self.ui_params.get("output_dir", "droppy_out")
                
                ga_falling = YLCurveFallingDroplet(
                    initial_sigma=self.inputs.sigma,
                    initial_canny1=self.inputs.canny1,
                    initial_plateau_length=self.ui_params["plateau_width"],
                    initial_low_clarity=self.ui_params.get("low_clarity_ratio", 0.1),
                    initial_high_clarity=self.ui_params.get("high_clarity_ratio", 0.01),
                    initial_bo=0.2,
                    pop_size=self.ui_params.get("ga_population", 8),
                    max_generations=self.ui_params.get("ga_generations", 30),
                )
                
                ga_best_params, ga_fitness_history = ga_falling.evolve(evaluate_params_falling, outdir_ga + "/ga_generations")
                ga_enabled_final = True
                
            # If GA was enabled, use its Bo result; otherwise use averaged Bo_low/Bo_high
            if ga_enabled_final and ga_best_params is not None:
                Bo_low = float(ga_best_params['bo'])
                Bo_high = float(ga_best_params['bo'])
                Bo_final = float(ga_best_params['bo'])
            else:
                Bo_low = float(result_low["best_Bo"])
                Bo_high = float(result_high["best_Bo"])
                Bo_final = 0.5 * (Bo_low + Bo_high)
            
            best = result_high
            
            # For rising droplets, check if YL curve intercepts needle
            yl_intercepts_needle = False
            yl_distance_to_needle = float('nan')
            
            if self.inputs.droplet_type == "rising" and np.isfinite(Bo_final):
                r_star_pred, z_star_pred = integrate_young_laplace(Bo_final, "rising", z_stop=3.0)
                intercepts, distance = check_yl_needle_interception(
                    r_star_pred, z_star_pred, R0_len_mm, tip, ex, ey, lensing_factor, mm_per_px
                )
                yl_intercepts_needle = intercepts
                yl_distance_to_needle = distance
                
                # If doesn't intercept, suggest auto-correction
                if not intercepts and distance > 3.0:
                    pass
            
            self.finished.emit({
                "success": True,
                "Bo_low": Bo_low,
                "Bo_high": Bo_high,
                "Bo_final": Bo_final,
                "best_error": best["best_error"],
                "rmse_med": best["rmse_med"],
                "rmse_thresh": best["rmse_thresh"],
                "rmse_med_mm": best["rmse_med_mm"],
                "rmse_thresh_mm": best["rmse_thresh_mm"],
                "yl_intercepts_needle": yl_intercepts_needle,
                "yl_distance_to_needle": yl_distance_to_needle,
                "ga_enabled": ga_enabled_final,
                "ga_best_params": ga_best_params,
                "droplet_type": self.inputs.droplet_type,
                "direction": self.inputs.direction,
                "needle_left_xy": self.inputs.needle_left_xy,
                "needle_right_xy": self.inputs.needle_right_xy,
                "plateau_mask": best["plateau_mask"],
                "plateau_length": best["plateau_length"],
                "img": img,
                "edges": edges,
                "pts_xy": pts_xy,
                "contour_xy_full": contour_xy_full,
                "tip": tip,
                "ctr": ctr,
                "lft": lft,
                "rht": rht,
                "ex": ex,
                "ey": ey,
                "px_per_mm": px_per_mm,
                "mm_per_px": mm_per_px,
                "R0_len_mm": R0_len_mm,
                "W_img": W_img,
                "H_img": H_img,
                "lensing_factor": lensing_factor,
            })
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tensiometry Analysis - GUI Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        self.image_path = None
        self.processing_thread = None
        
        self.init_ui()
        self.load_settings()  # Load settings after UI is initialized
    
    def init_ui(self):
        """Initialize UI."""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create title bar with logo and controls
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(5, 5, 5, 5)
        
        # Logo (left side) - try to load droppy.png
        self.logo_label = QLabel()
        try:
            logo_pixmap = QPixmap("droppy.png")
            if not logo_pixmap.isNull():
                scaled_logo = logo_pixmap.scaledToHeight(30, Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(scaled_logo)
            else:
                self.logo_label.setText("Droppy")
                self.logo_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        except Exception:
            self.logo_label.setText("Droppy")
            self.logo_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_layout.addWidget(self.logo_label)
        
        title_layout.addStretch()
        
        # Minimize and close buttons (right side)
        minimize_btn = QPushButton(" _ ")
        minimize_btn.setMaximumWidth(30)
        minimize_btn.setMaximumHeight(25)
        minimize_btn.setToolTip("Minimize")
        minimize_btn.setStyleSheet("""
            QPushButton { 
                background-color: #d3d3d3; 
                color: black; 
                font-weight: bold;
                border: 1px solid #999;
                padding: 0px;
            }
            QPushButton:hover { 
                background-color: #e0e0e0;
            }
        """)
        minimize_btn.clicked.connect(self.showMinimized)
        title_layout.addWidget(minimize_btn)
        
        close_btn = QPushButton(" ✕ ")
        close_btn.setMaximumWidth(30)
        close_btn.setMaximumHeight(25)
        close_btn.setToolTip("Close")
        close_btn.setStyleSheet("""
            QPushButton { 
                background-color: #d3d3d3; 
                color: red; 
                font-weight: bold;
                border: 1px solid #999;
                padding: 0px;
            }
            QPushButton:hover { 
                background-color: #ff6b6b;
                color: white;
            }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        
        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        title_widget.setStyleSheet("background-color: #f0f0f0; border-bottom: 1px solid #ccc;")
        main_layout.addWidget(title_widget)
        
        tabs = QTabWidget()
        tabs.addTab(self.create_image_tab(), "Image & Geometry")
        tabs.addTab(self.create_params_tab(), "Processing")
        tabs.addTab(self.create_analysis_tab(), "Analysis")
        tabs.addTab(self.create_results_tab(), "Results")
        
        main_layout.addWidget(tabs)
        
        run_layout = QHBoxLayout()
        
        # GA controls (checkbox + population/generations)
        self.use_genetic_algorithm = QCheckBox("Use GA Optimization")
        self.use_genetic_algorithm.setChecked(False)
        
        self.ga_population = QSpinBox()
        self.ga_population.setRange(4, 20)
        self.ga_population.setValue(8)
        self.ga_population.setMaximumWidth(60)
        
        self.ga_generations = QSpinBox()
        self.ga_generations.setRange(5, 100)
        self.ga_generations.setValue(30)
        self.ga_generations.setMaximumWidth(60)
        
        run_layout.addWidget(self.use_genetic_algorithm)
        run_layout.addWidget(QLabel("Pop:"))
        run_layout.addWidget(self.ga_population)
        run_layout.addWidget(QLabel("Gen:"))
        run_layout.addWidget(self.ga_generations)
        run_layout.addStretch()
        
        self.run_btn = QPushButton("RUN ANALYSIS")
        self.run_btn.setStyleSheet("""
            QPushButton { background-color: #2ecc71; color: white; font-weight: bold;
                padding: 12px; font-size: 14px; border-radius: 5px; }
            QPushButton:hover { background-color: #27ae60; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.run_btn.clicked.connect(self.on_run_analysis)
        self.run_btn.setEnabled(False)
        
        save_settings_btn = QPushButton(" Save Settings ")
        save_settings_btn.setMaximumWidth(120)
        save_settings_btn.setMaximumHeight(25)
        save_settings_btn.setMinimumWidth(25)
        save_settings_btn.setMinimumHeight(25)
        save_settings_btn.setToolTip("Save Settings")
        save_settings_btn.setStyleSheet("""
            QPushButton { 
                background-color: #d3d3d3; 
                color: black; 
                font-size: 10px;
                border: 2px outset #f0f0f0;
                border-top-color: #ffffff;
                border-left-color: #ffffff;
                border-bottom-color: #808080;
                border-right-color: #808080;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover { 
                background-color: #e0e0e0;
            }
            QPushButton:pressed { 
                border: 2px inset #f0f0f0;
                border-top-color: #808080;
                border-left-color: #808080;
                border-bottom-color: #ffffff;
                border-right-color: #ffffff;
            }
        """)
        save_settings_btn.clicked.connect(self.on_save_settings)
        
        run_layout.addWidget(self.run_btn)
        run_layout.addStretch()
        
        help_btn = QPushButton("?")
        help_btn.setMaximumWidth(25)
        help_btn.setMaximumHeight(25)
        help_btn.setMinimumWidth(25)
        help_btn.setMinimumHeight(25)
        help_btn.setToolTip("Help")
        help_btn.setStyleSheet("""
            QPushButton { 
                background-color: #d3d3d3; 
                color: black; 
                font-weight: bold;
                font-size: 12px;
                border: 2px outset #f0f0f0;
                border-top-color: #ffffff;
                border-left-color: #ffffff;
                border-bottom-color: #808080;
                border-right-color: #808080;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover { 
                background-color: #e0e0e0;
            }
            QPushButton:pressed { 
                border: 2px inset #f0f0f0;
                border-top-color: #808080;
                border-left-color: #808080;
                border-bottom-color: #ffffff;
                border-right-color: #ffffff;
            }
        """)
        help_btn.clicked.connect(self.on_help)
        
        # Group Save Settings and Help buttons together on the right
        right_buttons_layout = QHBoxLayout()
        right_buttons_layout.setContentsMargins(0, 0, 0, 0)
        right_buttons_layout.setSpacing(5)
        right_buttons_layout.addWidget(save_settings_btn)
        right_buttons_layout.addWidget(help_btn)
        
        right_buttons_widget = QWidget()
        right_buttons_widget.setLayout(right_buttons_layout)
        run_layout.addWidget(right_buttons_widget)
        
        main_layout.addLayout(run_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def create_image_tab(self) -> QWidget:
        """Image and geometry tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Image:"))
        self.image_label = QLineEdit()
        self.image_label.setReadOnly(True)
        img_layout.addWidget(self.image_label)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.on_browse_image)
        img_layout.addWidget(browse_btn)
        layout.addLayout(img_layout)
        
        geo_group = QGroupBox("Needle & ROI")
        geo_layout = QFormLayout()
        
        self.needle_left_x = QDoubleSpinBox()
        self.needle_left_x.setRange(0, 10000)
        self.needle_left_y = QDoubleSpinBox()
        self.needle_left_y.setRange(0, 10000)
        self.needle_right_x = QDoubleSpinBox()
        self.needle_right_x.setRange(0, 10000)
        self.needle_right_y = QDoubleSpinBox()
        self.needle_right_y.setRange(0, 10000)
        
        needle_left_layout = QHBoxLayout()
        needle_left_layout.addWidget(self.needle_left_x)
        needle_left_layout.addWidget(self.needle_left_y)
        geo_layout.addRow("Needle Left (X, Y):", needle_left_layout)
        
        needle_right_layout = QHBoxLayout()
        needle_right_layout.addWidget(self.needle_right_x)
        needle_right_layout.addWidget(self.needle_right_y)
        geo_layout.addRow("Needle Right (X, Y):", needle_right_layout)
        
        self.needle_diameter = QDoubleSpinBox()
        self.needle_diameter.setRange(0.01, 10.0)
        self.needle_diameter.setValue(0.72)
        self.needle_diameter.setSingleStep(0.01)
        geo_layout.addRow("Needle Diameter (mm):", self.needle_diameter)
        
        self.droplet_type = QComboBox()
        self.droplet_type.addItems(["rising", "pendant"])
        geo_layout.addRow("Droplet Type:", self.droplet_type)
        
        self.roi_x = QSpinBox()
        self.roi_x.setRange(0, 10000)
        self.roi_y = QSpinBox()
        self.roi_y.setRange(0, 10000)
        self.roi_w = QSpinBox()
        self.roi_w.setRange(1, 10000)
        self.roi_w.setValue(500)
        self.roi_h = QSpinBox()
        self.roi_h.setRange(1, 10000)
        self.roi_h.setValue(500)
        
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("X:"))
        roi_layout.addWidget(self.roi_x)
        roi_layout.addWidget(QLabel("Y:"))
        roi_layout.addWidget(self.roi_y)
        roi_layout.addWidget(QLabel("W:"))
        roi_layout.addWidget(self.roi_w)
        roi_layout.addWidget(QLabel("H:"))
        roi_layout.addWidget(self.roi_h)
        geo_layout.addRow("ROI:", roi_layout)
        
        geo_group.setLayout(geo_layout)
        layout.addWidget(geo_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_params_tab(self) -> QWidget:
        """Processing parameters."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        img_group = QGroupBox("Image Processing")
        img_layout = QFormLayout()
        
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setValue(2.0)
        img_layout.addRow("Blur Sigma:", self.sigma)
        
        self.canny1 = QSpinBox()
        self.canny1.setRange(1, 500)
        self.canny1.setValue(40)
        img_layout.addRow("Canny Low:", self.canny1)
        
        self.canny2 = QSpinBox()
        self.canny2.setRange(1, 500)
        self.canny2.setValue(120)
        img_layout.addRow("Canny High:", self.canny2)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        contour_group = QGroupBox("Contour")
        contour_layout = QFormLayout()
        
        self.num_points = QSpinBox()
        self.num_points.setRange(10, 10000)
        self.num_points.setValue(450)
        contour_layout.addRow("Num Points:", self.num_points)
        
        self.direction = QSpinBox()
        self.direction.setRange(-1, 1)
        self.direction.setValue(1)
        contour_layout.addRow("Direction:", self.direction)
        
        contour_group.setLayout(contour_layout)
        layout.addWidget(contour_group)
        
        num_group = QGroupBox("Thresholds")
        num_layout = QFormLayout()
        
        self.min_r_mm = QDoubleSpinBox()
        self.min_r_mm.setRange(0.001, 10.0)
        self.min_r_mm.setValue(0.08)
        num_layout.addRow("Min r' (mm):", self.min_r_mm)
        
        self.min_z_mm = QDoubleSpinBox()
        self.min_z_mm.setRange(0.001, 10.0)
        self.min_z_mm.setValue(0.08)
        num_layout.addRow("Min z' (mm):", self.min_z_mm)
        
        self.deriv_window = QSpinBox()
        self.deriv_window.setRange(1, 50)
        self.deriv_window.setValue(6)
        num_layout.addRow("Deriv Window:", self.deriv_window)
        
        self.rmse_factor = QDoubleSpinBox()
        self.rmse_factor.setRange(0.1, 10.0)
        self.rmse_factor.setValue(3.0)
        num_layout.addRow("RMSE Factor:", self.rmse_factor)
        
        self.mad_z = QDoubleSpinBox()
        self.mad_z.setRange(0.1, 10.0)
        self.mad_z.setValue(3.5)
        num_layout.addRow("MAD Z:", self.mad_z)
        
        num_group.setLayout(num_layout)
        layout.addWidget(num_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_analysis_tab(self) -> QWidget:
        """Analysis parameters."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        stable_group = QGroupBox("Stability")
        stable_layout = QFormLayout()
        
        self.stable_s_min_frac = QDoubleSpinBox()
        self.stable_s_min_frac.setRange(0.0, 1.0)
        self.stable_s_min_frac.setValue(0.0)
        stable_layout.addRow("Min Arc Frac:", self.stable_s_min_frac)
        
        self.stable_s_max_frac = QDoubleSpinBox()
        self.stable_s_max_frac.setRange(0.0, 1.0)
        self.stable_s_max_frac.setValue(1.0)
        stable_layout.addRow("Max Arc Frac:", self.stable_s_max_frac)
        
        stable_group.setLayout(stable_layout)
        layout.addWidget(stable_group)
        
        bo_group = QGroupBox("Bond Number")
        bo_layout = QFormLayout()
        
        self.plateau_width = QSpinBox()
        self.plateau_width.setRange(10, 500)
        self.plateau_width.setValue(100)
        bo_layout.addRow("Plateau Width:", self.plateau_width)
        
        self.bo_wiggle_room = QDoubleSpinBox()
        self.bo_wiggle_room.setRange(0.0, 10.0)
        self.bo_wiggle_room.setValue(0.0)
        bo_layout.addRow("Bo Wiggle Room:", self.bo_wiggle_room)
        
        self.lensing_factor = QDoubleSpinBox()
        self.lensing_factor.setRange(0.1, 3.0)
        self.lensing_factor.setValue(1.0)
        self.lensing_factor.setSingleStep(0.001)
        self.lensing_factor.setDecimals(3)
        bo_layout.addRow("Lensing Factor:", self.lensing_factor)
        
        bo_group.setLayout(bo_layout)
        layout.addWidget(bo_group)
        
        clarity_group = QGroupBox("Clarity Ratios")
        clarity_layout = QFormLayout()
        
        self.low_clarity_ratio = QDoubleSpinBox()
        self.low_clarity_ratio.setRange(0.001, 1.0)
        self.low_clarity_ratio.setValue(0.1)
        clarity_layout.addRow("Low Clarity:", self.low_clarity_ratio)
        
        self.high_clarity_ratio = QDoubleSpinBox()
        self.high_clarity_ratio.setRange(0.001, 1.0)
        self.high_clarity_ratio.setValue(0.01)
        clarity_layout.addRow("High Clarity:", self.high_clarity_ratio)
        
        clarity_group.setLayout(clarity_layout)
        
        layout.addWidget(clarity_group)
        

        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_results_tab(self) -> QWidget:
        """Results display."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText("Results will appear here...")
        layout.addWidget(self.results_text)
        
        layout.addWidget(QLabel("Visualizations:"))
        
        img_layout = QHBoxLayout()
        
        img_layout.addWidget(QLabel("YL Overlay:"))
        self.viz_label = QLabel()
        self.viz_label.setMinimumSize(350, 350)
        self.viz_label.setStyleSheet("border: 1px solid gray;")
        self.viz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_label.setText("(YL fit overlay)")
        img_layout.addWidget(self.viz_label)
        
        img_layout.addWidget(QLabel("Edge Detection:"))
        self.edges_label = QLabel()
        self.edges_label.setMinimumSize(350, 350)
        self.edges_label.setStyleSheet("border: 1px solid gray;")
        self.edges_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edges_label.setText("(Edge detection)")
        img_layout.addWidget(self.edges_label)
        
        layout.addLayout(img_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Dir:"))
        self.output_dir = QLineEdit()
        self.output_dir.setText("droplet_out")
        output_layout.addWidget(self.output_dir)
        browse_out = QPushButton("Browse...")
        browse_out.clicked.connect(self.on_browse_output)
        output_layout.addWidget(browse_out)
        layout.addLayout(output_layout)
        
        widget.setLayout(layout)
        return widget
    
    def on_browse_image(self):
        """Select image and launch geometry selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if file_path:
            self.image_path = file_path
            self.image_label.setText(file_path)
            
            try:
                geometry = self.interactive_geometry_selection(file_path)
                if geometry:
                    self.droplet_type.setCurrentText(geometry["droplet_type"])
                    self.needle_left_x.setValue(geometry["needle_left_xy"][0])
                    self.needle_left_y.setValue(geometry["needle_left_xy"][1])
                    self.needle_right_x.setValue(geometry["needle_right_xy"][0])
                    self.needle_right_y.setValue(geometry["needle_right_xy"][1])
                    self.needle_diameter.setValue(geometry["needle_diameter_mm"])
                    self.roi_x.setValue(geometry["roi_box"][0])
                    self.roi_y.setValue(geometry["roi_box"][1])
                    self.roi_w.setValue(geometry["roi_box"][2])
                    self.roi_h.setValue(geometry["roi_box"][3])
                    self.run_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Geometry selection failed: {str(e)}")
    
    def interactive_geometry_selection(self, image_path: str) -> Optional[dict]:
        """Interactive cv2 geometry selection."""
        img_full = cv2.imread(image_path)
        if img_full is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        QMessageBox.information(self, "Step 1", "Draw ROI box around droplet, then press ENTER")
        
        VIEW_W, VIEW_H = 500, 500
        H, W = img_full.shape[:2]
        vx = max(0, (W - VIEW_W) // 2)
        vy = max(0, (H - VIEW_H) // 2)
        
        mode = "roi"
        drawing = False
        start_pt = None
        current_pt = None
        roi_box = None
        needle_line = None
        
        def get_view():
            return img_full[vy:vy + VIEW_H, vx:vx + VIEW_W].copy()
        
        def mouse(event, x, y, flags, param):
            nonlocal drawing, start_pt, current_pt, roi_box, needle_line
            full_x, full_y = vx + x, vy + y
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing, start_pt, current_pt = True, (full_x, full_y), (full_x, full_y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                current_pt = (full_x, full_y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if mode == "roi":
                    roi_box = (start_pt, (full_x, full_y))
                elif mode == "needle":
                    needle_line = (start_pt, (full_x, full_y))
        
        cv2.namedWindow("Geometry Selection")
        cv2.setMouseCallback("Geometry Selection", mouse)
        
        while True:
            view = get_view()
            if drawing and start_pt and current_pt:
                if mode == "roi":
                    cv2.rectangle(view, (start_pt[0]-vx, start_pt[1]-vy), (current_pt[0]-vx, current_pt[1]-vy), (0,255,0), 2)
                elif mode == "needle":
                    cv2.line(view, (start_pt[0]-vx, start_pt[1]-vy), (current_pt[0]-vx, current_pt[1]-vy), (0,255,255), 2)
            if roi_box:
                cv2.rectangle(view, (roi_box[0][0]-vx, roi_box[0][1]-vy), (roi_box[1][0]-vx, roi_box[1][1]-vy), (0,255,0), 2)
            if needle_line:
                cv2.line(view, (needle_line[0][0]-vx, needle_line[0][1]-vy), (needle_line[1][0]-vx, needle_line[1][1]-vy), (0,255,255), 2)
            
            cv2.imshow("Geometry Selection", view)
            key = cv2.waitKey(20)
            
            if key == 81: vx = max(0, vx - 20)
            elif key == 83: vx = min(W - VIEW_W, vx + 20)
            elif key == 82: vy = max(0, vy - 20)
            elif key == 84: vy = min(H - VIEW_H, vy + 20)
            elif key == 13:
                if mode == "roi" and roi_box:
                    mode = "needle"
                    cv2.destroyWindow("Geometry Selection")
                    QMessageBox.information(self, "Step 2", "Draw needle line, then press ENTER")
                    cv2.namedWindow("Geometry Selection")
                    cv2.setMouseCallback("Geometry Selection", mouse)
                elif mode == "needle" and needle_line:
                    break
            elif key == 27:
                cv2.destroyWindow("Geometry Selection")
                return None
        
        cv2.destroyWindow("Geometry Selection")
        
        (rx1, ry1), (rx2, ry2) = roi_box
        x0, y0 = int(min(rx1, rx2)), int(min(ry1, ry2))
        w, h = int(abs(rx2-rx1)), int(abs(ry2-ry1))
        
        needle_diameter_mm = self.prompt_needle_diameter()
        if needle_diameter_mm is None:
            return None
        
        droplet_type = self.prompt_droplet_type()
        if droplet_type is None:
            return None
        
        return {
            "droplet_type": droplet_type,
            "needle_left_xy": needle_line[0],
            "needle_right_xy": needle_line[1],
            "needle_diameter_mm": needle_diameter_mm,
            "roi_box": (x0, y0, w, h),
        }
    
    def prompt_needle_diameter(self) -> Optional[float]:
        """Prompt for needle diameter."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Needle Diameter")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter needle diameter (mm):"))
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.01, 10.0)
        spinbox.setValue(0.72)
        layout.addWidget(spinbox)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        
        result = [None]
        
        def on_ok():
            result[0] = spinbox.value()
            dialog.accept()
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
        return result[0]
    
    def prompt_droplet_type(self) -> Optional[str]:
        """Prompt for droplet type."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Droplet Type")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select droplet type:"))
        combo = QComboBox()
        combo.addItems(["rising", "pendant"])
        layout.addWidget(combo)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        
        result = [None]
        
        def on_ok():
            result[0] = combo.currentText()
            dialog.accept()
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
        return result[0]
    
    def on_browse_output(self):
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir.setText(dir_path)
    
    def on_help(self):
        """Open GitHub documentation."""
        import webbrowser
        webbrowser.open("https://github.com/pendantdroppy/pendantdroppy")
    
    def on_save_settings(self):
        """Save current settings to droppy.conf file."""
        outdir = "/home/"+user+"/.droppy/"
        if not outdir:
            QMessageBox.warning(self, "Error", "Please specify an output directory first")
            return
        
        os.makedirs(outdir, exist_ok=True)
        
        settings = {
            "image_path": self.image_path if self.image_path else "",
            "droplet_type": self.droplet_type.currentText(),
            "needle_left_xy": [self.needle_left_x.value(), self.needle_left_y.value()],
            "needle_right_xy": [self.needle_right_x.value(), self.needle_right_y.value()],
            "needle_diameter_mm": self.needle_diameter.value(),
            "roi": [self.roi_x.value(), self.roi_y.value(), self.roi_w.value(), self.roi_h.value()],
            "sigma": self.sigma.value(),
            "canny1": self.canny1.value(),
            "canny2": self.canny2.value(),
            "num_points": self.num_points.value(),
            "direction": self.direction.value(),
            "min_r_mm": self.min_r_mm.value(),
            "min_z_mm": self.min_z_mm.value(),
            "deriv_window": self.deriv_window.value(),
            "stable_s_min_frac": self.stable_s_min_frac.value(),
            "stable_s_max_frac": self.stable_s_max_frac.value(),
            "rmse_factor": self.rmse_factor.value(),
            "mad_z": self.mad_z.value(),
            "plateau_width": self.plateau_width.value(),
            "bo_wiggle_room": self.bo_wiggle_room.value(),
            "lensing_factor": self.lensing_factor.value(),
            "low_clarity_ratio": self.low_clarity_ratio.value(),
            "high_clarity_ratio": self.high_clarity_ratio.value(),
        }
        
        conf_path = os.path.join(outdir, "droppy.conf")
        try:
            with open(conf_path, "w") as f:
                json.dump(settings, f, indent=2)
            QMessageBox.information(self, "Success", f"Settings saved to:\n{conf_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{str(e)}")
    
    def load_settings(self):
        """Load settings from droppy.conf if it exists, otherwise use defaults."""
        settings_outdir = "/home/"+user+"/.droppy/"
        default_outdir = "/home/"+user+"/.droppy/droplet_out"
        conf_path = os.path.join(settings_outdir, "droppy.conf")
        
        if os.path.exists(conf_path):
            try:
                with open(conf_path, "r") as f:
                    settings = json.load(f)
                
                # Load all settings with safe defaults
                if settings.get("image_path"):
                    self.image_path = settings["image_path"]
                    self.image_label.setText(self.image_path)
                
                self.droplet_type.setCurrentText(settings.get("droplet_type", "rising"))
                
                needle_left = settings.get("needle_left_xy", [0, 0])
                self.needle_left_x.setValue(needle_left[0])
                self.needle_left_y.setValue(needle_left[1])
                
                needle_right = settings.get("needle_right_xy", [0, 0])
                self.needle_right_x.setValue(needle_right[0])
                self.needle_right_y.setValue(needle_right[1])
                
                self.needle_diameter.setValue(settings.get("needle_diameter_mm", 0.72))
                
                roi = settings.get("roi", [0, 0, 500, 500])
                self.roi_x.setValue(roi[0])
                self.roi_y.setValue(roi[1])
                self.roi_w.setValue(roi[2])
                self.roi_h.setValue(roi[3])
                
                self.sigma.setValue(settings.get("sigma", 2.0))
                self.canny1.setValue(settings.get("canny1", 40))
                self.canny2.setValue(settings.get("canny2", 120))
                self.num_points.setValue(settings.get("num_points", 450))
                self.direction.setValue(settings.get("direction", 1))
                self.min_r_mm.setValue(settings.get("min_r_mm", 0.08))
                self.min_z_mm.setValue(settings.get("min_z_mm", 0.08))
                self.deriv_window.setValue(settings.get("deriv_window", 6))
                self.stable_s_min_frac.setValue(settings.get("stable_s_min_frac", 0.0))
                self.stable_s_max_frac.setValue(settings.get("stable_s_max_frac", 1.0))
                self.rmse_factor.setValue(settings.get("rmse_factor", 3.0))
                self.mad_z.setValue(settings.get("mad_z", 3.5))
                self.plateau_width.setValue(settings.get("plateau_width", 100))
                self.bo_wiggle_room.setValue(settings.get("bo_wiggle_room", 0.0))
                self.lensing_factor.setValue(settings.get("lensing_factor", 1.0))
                self.low_clarity_ratio.setValue(settings.get("low_clarity_ratio", 0.1))
                self.high_clarity_ratio.setValue(settings.get("high_clarity_ratio", 0.01))
                
                self.output_dir.setText(default_outdir)
                
            except Exception as e:
                # Failed to load, use defaults
                self.output_dir.setText(default_outdir)
        else:
            # No config file, use defaults
            self.output_dir.setText(default_outdir)
    
    def on_run_analysis(self):
        """Run analysis."""
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Select image first")
            return
        
        try:
            inputs = Inputs(
                image_path=self.image_path,
                droplet_type=self.droplet_type.currentText(),
                needle_left_xy=(self.needle_left_x.value(), self.needle_left_y.value()),
                needle_right_xy=(self.needle_right_x.value(), self.needle_right_y.value()),
                needle_diameter_mm=self.needle_diameter.value(),
                sigma=self.sigma.value(),
                canny1=self.canny1.value(),
                canny2=self.canny2.value(),
                num_points=self.num_points.value(),
                direction=self.direction.value(),
                min_r_mm=self.min_r_mm.value(),
                min_z_mm=self.min_z_mm.value(),
                circle_window=max(1, int(self.high_clarity_ratio.value() * self.num_points.value())),
                deriv_window=self.deriv_window.value(),
                stable_s_min_frac=self.stable_s_min_frac.value(),
                stable_s_max_frac=self.stable_s_max_frac.value(),
                rmse_factor=self.rmse_factor.value(),
                mad_z=self.mad_z.value(),
                circle_window_frac=self.high_clarity_ratio.value(),
            )
            
            ui_params = {
                "roi_box": (self.roi_x.value(), self.roi_y.value(), self.roi_w.value(), self.roi_h.value()),
                "plateau_width": self.plateau_width.value(),
                "bo_wiggle_room": self.bo_wiggle_room.value(),
                "lensing_factor": self.lensing_factor.value(),
                "low_clarity_ratio": self.low_clarity_ratio.value(),
                "high_clarity_ratio": self.high_clarity_ratio.value(),
                "use_genetic_algorithm": self.use_genetic_algorithm.isChecked(),
                "ga_population": self.ga_population.value(),
                "ga_generations": self.ga_generations.value(),
            }
            
            self.run_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.processing_thread = ProcessingThread(inputs, ui_params)
            self.processing_thread.progress.connect(self.on_progress)
            self.processing_thread.finished.connect(self.on_finished)
            self.processing_thread.error.connect(self.on_error)
            self.processing_thread.start()
            
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid parameters: {str(e)}")
    
    def on_progress(self, message: str):
        """Update progress."""
        self.status_label.setText(message)
        self.progress_bar.setValue(min(95, self.progress_bar.value() + 10))
    
    def on_finished(self, result: dict):
        """Handle completion."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if result["success"]:
            # Extract results
            Bo_low = result["Bo_low"]
            Bo_high = result["Bo_high"]
            Bo_final = result["Bo_final"]
            
            # Create output directory
            outdir = self.output_dir.text()
            os.makedirs(outdir, exist_ok=True)
            
            # Extract geometry
            img = result["img"]
            edges = result["edges"]
            pts_xy = np.array(result["pts_xy"])
            tip = np.array(result["tip"])
            ctr = np.array(result["ctr"])
            lft = np.array(result["lft"])
            rht = np.array(result["rht"])
            ex = np.array(result["ex"])
            ey = np.array(result["ey"])
            plateau_mask = np.array(result["plateau_mask"])
            
            px_per_mm = result["px_per_mm"]
            mm_per_px = result["mm_per_px"]
            R0_len_mm = result["R0_len_mm"]
            W_img = result["W_img"]
            H_img = result["H_img"]
            lensing_factor = result["lensing_factor"]
            
            # Get ROI bounds for cropping - EXPAND BY 10%
            roi_x_input = self.roi_x.value()
            roi_y_input = self.roi_y.value()
            roi_w_input = self.roi_w.value()
            roi_h_input = self.roi_h.value()
            
            # Expand by 10% on all sides
            expand_x = int(roi_w_input * 0.05)
            expand_y = int(roi_h_input * 0.05)
            
            x0 = max(0, roi_x_input - expand_x)
            y0 = max(0, roi_y_input - expand_y)
            x1 = min(W_img, roi_x_input + roi_w_input + expand_x)
            y1 = min(H_img, roi_y_input + roi_h_input + expand_y)
            
            rw = x1 - x0
            rh = y1 - y0
            
            # Crop image to expanded ROI
            viz_img = img[y0:y1, x0:x1].copy()
            edges_roi = edges[y0:y1, x0:x1]
            
            # Draw YL curve in red - USE AVERAGED BO, then scale for visual lensing correction
            if np.isfinite(Bo_final):
                r_star_pred, z_star_pred = integrate_young_laplace(Bo_final, self.droplet_type.currentText(), z_stop=3.0)
                if len(r_star_pred) > 10:
                    r_mm_pred = r_star_pred * R0_len_mm
                    z_mm_pred = z_star_pred * R0_len_mm
                    r_px_pred = r_mm_pred / mm_per_px  # No lensing division - it's in r_star already
                    z_px_pred = z_mm_pred / mm_per_px
                    
                    # APPLY LENSING FACTOR FOR VISUALIZATION ONLY
                    # If lensing_factor < 1, stretch horizontally (divide r_px to move further out)
                    # If lensing_factor > 1, compress horizontally (multiply r_px to move closer in)
                    r_px_pred_visual = r_px_pred / lensing_factor
                    
                    for i in range(len(r_px_pred_visual) - 1):
                        pt1 = (tip + r_px_pred_visual[i] * ex + z_px_pred[i] * ey).astype(int)
                        pt2 = (tip + r_px_pred_visual[i+1] * ex + z_px_pred[i+1] * ey).astype(int)
                        pt1_roi = (pt1[0] - x0, pt1[1] - y0)
                        pt2_roi = (pt2[0] - x0, pt2[1] - y0)
                        if (0 <= pt1_roi[0] < rw and 0 <= pt1_roi[1] < rh and
                            0 <= pt2_roi[0] < rw and 0 <= pt2_roi[1] < rh):
                            cv2.line(viz_img, pt1_roi, pt2_roi, (0, 0, 255), 2)
            
            # Draw plateau in yellow
            for i, p in enumerate(pts_xy):
                if plateau_mask[i]:
                    p_roi = (int(p[0] - x0), int(p[1] - y0))
                    if 0 <= p_roi[0] < rw and 0 <= p_roi[1] < rh:
                        cv2.circle(viz_img, p_roi, 3, (0, 255, 255), -1)
            
            # Draw extrema in gray
            for point in [tip, ctr, lft, rht]:
                p_roi = (int(point[0] - x0), int(point[1] - y0))
                if 0 <= p_roi[0] < rw and 0 <= p_roi[1] < rh:
                    cv2.circle(viz_img, p_roi, 6, (128, 128, 128), -1)
            
            # For rising droplets, add needle corner marker in traversal direction
            droplet_type = result.get("droplet_type", "pendant")
            direction = result.get("direction", 1)
            
            if droplet_type == "rising":
                # Mark the LEFT end of the needle (the corner where droplet contacts)
                # needle_left_xy is the left edge of the needle
                needle_corner_screen = np.array(result.get("needle_left_xy", [0, 0]), dtype=np.float64)
                nc_roi = (int(needle_corner_screen[0] - x0), int(needle_corner_screen[1] - y0))
                if 0 <= nc_roi[0] < rw and 0 <= nc_roi[1] < rh:
                    cv2.circle(viz_img, nc_roi, 5, (0, 255, 255), -1)  # Cyan - very visible
            
            # Save files
            cv2.imwrite(os.path.join(outdir, "result_yl_overlay.png"), viz_img)
            cv2.imwrite(os.path.join(outdir, "edges.png"), cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR))
            
            # Save summary
            summary = {
                "Bo_low": Bo_low,
                "Bo_high": Bo_high,
                "Bo_final": Bo_final,
                "plateau_length": result["plateau_length"],
                "R0_len_mm": R0_len_mm,
                "px_per_mm": px_per_mm,
                "mm_per_px": mm_per_px,
                "lensing_factor": lensing_factor,
                "rmse_median_mm": result["rmse_med_mm"],
                "rmse_threshold_mm": result["rmse_thresh_mm"],
                "rmse_median_px": result["rmse_med"],
                "rmse_threshold_px": result["rmse_thresh"],
                "yl_intercepts_needle": bool(result["yl_intercepts_needle"]),
                "yl_distance_to_needle": float(result["yl_distance_to_needle"]),
                "ga_enabled": bool(result["ga_enabled"]),
                "ga_best_params": {
                    "sigma": float(result["ga_best_params"]["sigma"]),
                    "canny1": int(result["ga_best_params"]["canny1"]),
                    "plateau_length": int(result["ga_best_params"]["plateau_length"]),
                } if result["ga_best_params"] else None,
            }
            with open(os.path.join(outdir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            
            # Display results
            self.display_results(viz_img, edges_roi, summary, outdir)
            self.status_label.setText("Analysis complete!")
            
            QMessageBox.information(
                self, "Success",
                f"Analysis complete!\n\n"
                f"Bo (low):  {Bo_low:.6f}\n"
                f"Bo (high): {Bo_high:.6f}\n"
                f"Bo (avg):  {Bo_final:.6f}\n\n"
                f"Output: {outdir}"
            )
        else:
            self.status_label.setText("✗ Analysis failed")
    
    def display_results(self, viz_img, edges_img, summary, outdir):
        """Display results in Qt."""
        viz_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
        edges_rgb = cv2.cvtColor(cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        
        h, w = viz_rgb.shape[:2]
        bytes_per_line = 3 * w
        
        qt_viz = QImage(viz_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        qt_edges = QImage(edges_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        ga_params_str = ""
        if summary.get("ga_enabled", False) and summary.get("ga_best_params"):
            params = summary["ga_best_params"]
            ga_params_str = f"""
GENETIC ALGORITHM OPTIMIZATION:
  Status:            ENABLED & COMPLETED
  Optimized sigma:   {params.get('sigma', 'N/A'):.3f}
  Optimized Canny1:  {params.get('canny1', 'N/A')}
  Optimized Plateau: {params.get('plateau_length', 'N/A')} points
"""
        
        self.results_text.setText(f"""

RESULTS

CALIBRATION:
  • R0 (mm):           {summary['R0_len_mm']:.6f}
  • px/mm:             {summary['px_per_mm']:.6f}
  • mm/px:             {summary['mm_per_px']:.6f}

BOND NUMBERS:
  • Low Clarity:   {summary['Bo_low']:.6f}
  • High Clarity:  {summary['Bo_high']:.6f}
  • Averaged:      {summary['Bo_final']:.6f}

YL CURVE VALIDATION (Rising Droplets):
  Intercepts Needle: {('YES' if summary['yl_intercepts_needle'] else 'NO')}
  Distance to Needle: {summary['yl_distance_to_needle']:.2f} px
  Status: {'Good fit' if summary['yl_intercepts_needle'] else 'Check parameters'}

RMS ERROR (Circle Detection Quality):
  • Median RMSE:       {summary['rmse_median_mm']:.6f} mm
  • RMSE Threshold:    {summary['rmse_threshold_mm']:.6f} mm
  • (Median RMSE:      {summary['rmse_median_px']:.2f} px)
  • (Threshold:        {summary['rmse_threshold_px']:.2f} px)

PLATEAU:
  • Length:            {summary['plateau_length']} points

LENSING:
  • Factor:            {summary['lensing_factor']:.3f}
{ga_params_str}
Location: {outdir}
""")
        
        pixmap_viz = QPixmap.fromImage(qt_viz)
        pixmap_edges = QPixmap.fromImage(qt_edges)
        
        scaled_viz = pixmap_viz.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
        scaled_edges = pixmap_edges.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
        
        self.viz_label.setPixmap(scaled_viz)
        self.edges_label.setPixmap(scaled_edges)
    
    def on_error(self, error_msg: str):
        """Handle error."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("✗ Error")
        QMessageBox.critical(self, "Error", error_msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
