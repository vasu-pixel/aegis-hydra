
import ctypes
import os
import time
import numpy as np
from typing import Tuple

# Load Shared Library
LIB_PATH = os.path.join(os.path.dirname(__file__), "../cpp/libising.so")

class CppIsingGrid:
    """
    High-Performance Python Wrapper for C++ Ising Kernel.
    """
    def __init__(self, size: int, seed: int = None):
        self.size = size
        self.height = size
        self.width = size
        
        if seed is None:
            seed = int(time.time())
            
        # Load Library
        if not os.path.exists(LIB_PATH):
            raise FileNotFoundError(f"Compiler Error: {LIB_PATH} not found. Run 'make' in aegis_hydra/cpp/.")
            
        self.lib = ctypes.CDLL(LIB_PATH)
        
        # Define Argument Types
        self.lib.Ising_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
        self.lib.Ising_new.restype = ctypes.c_void_p
        
        self.lib.Ising_delete.argtypes = [ctypes.c_void_p]
        self.lib.Ising_delete.restype = None
        
        self.lib.Ising_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.Ising_step.restype = None
        
        self.lib.Ising_magnetization.argtypes = [ctypes.c_void_p]
        self.lib.Ising_magnetization.restype = ctypes.c_float
        
        self.lib.Ising_get_spins.argtypes = [ctypes.c_void_p]
        self.lib.Ising_get_spins.restype = ctypes.POINTER(ctypes.c_int8)
        
        # Initialize C++ Model
        self.model = self.lib.Ising_new(self.height, self.width, seed)
        
    def __del__(self):
        if hasattr(self, 'model') and self.model:
            self.lib.Ising_delete(self.model)
            
    def step(self, T: float, J: float, h: float) -> float:
        """
        Advance physics by one Monte Carlo sweep.
        Returns: Magnetization (float)
        """
        self.lib.Ising_step(self.model, T, J, h)
        return self.lib.Ising_magnetization(self.model)
        
    def get_spins(self) -> np.ndarray:
        """
        Get current spin configuration as numpy array.
        Zero-copy if possible, but ctypes pointers are tricky.
        Here we copy for safety.
        """
        ptr = self.lib.Ising_get_spins(self.model)
        # Create numpy array from pointer
        buffer = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int8 * (self.height * self.width)))
        arr = np.frombuffer(buffer.contents, dtype=np.int8)
        return arr.reshape((self.height, self.width)).copy()

# ==========================================
# PHASE 11: Real-Time Engine (Background C++)
# ==========================================
ENGINE_LIB_PATH = os.path.join(os.path.dirname(__file__), "../cpp/libising_engine.so")

class CppEngine:
    """
    Control interface for the background C++ Physics Engine.
    """
    def __init__(self):
        if not os.path.exists(ENGINE_LIB_PATH):
            raise FileNotFoundError(f"Compiler Error: {ENGINE_LIB_PATH} not found. Run 'make' in aegis_hydra/cpp/.")
            
        self.lib = ctypes.CDLL(ENGINE_LIB_PATH)
        
        self.lib.Engine_start.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
        self.lib.Engine_stop.restype = None
        
        self.lib.Engine_update_market.argtypes = [ctypes.c_float]
        self.lib.Engine_update_params.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        
        self.lib.Engine_get_magnetization.restype = ctypes.c_float
        self.lib.Engine_get_steps.restype = ctypes.c_long

    def start(self, size: int, seed: int = None):
        if seed is None: seed = int(time.time())
        self.lib.Engine_start(size, size, seed)
        
    def stop(self):
        self.lib.Engine_stop()
        
    def update_market(self, price: float):
        self.lib.Engine_update_market(price)
        
    def update_params(self, T: float, J: float, h: float):
        self.lib.Engine_update_params(T, J, h)
        
    def get_magnetization(self) -> float:
        return self.lib.Engine_get_magnetization()
        
    def get_steps(self) -> int:
        return self.lib.Engine_get_steps()
