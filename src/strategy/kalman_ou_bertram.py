from src.types import Signal, Tick
from src.strategy.base import BaseStrategy

from datetime import datetime
from collections import deque
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
from scipy.integrate import quad
from enum import Enum

class State(Enum):
    Short = "SHORT", 
    Long = "LONG", 
    Flat = "FLAT"

class KalmanOuBertram(BaseStrategy):
    
    
    def __init__(self, tickers, refit_interval, buffer_size, min_buffer, cost):
        
        # Should this be cast to a set for quicker lookup?
        # But also we need to verify that tickers here is only two different things -> can't have functionality for more
        self.tickers = tickers
        self.asset_a = self.tickers[0]
        self.asset_b = self.tickers[1]
        
        self.refit_interval = refit_interval
        self.buffer_size = buffer_size
        self.min_buffer = min_buffer
        
        # Data Management
        self.latest_prices = {
            ticker : None
            for ticker in tickers
        }
        self.fresh = set[str]
        self.latest_timestamp = None
        
        # Kalman
        self.beta = 1.00 # this should probably be an input to the function
        self.P = 1.00 # same for this
        self.delta= 1e-5 # same for this
        self.r = 1e-3 # same for this
        
        self.spread_buffer = deque(maxlen = buffer_size)
        
        # OU
        self.dt = 1/252
        self.sigma = None
        self.mu = None
        self.theta = None
        
        # Bertram
        self.cost = cost
        self.long_threshold = None
        self.short_threshold = None
        
        # Internal States
        
        self.curr_interval = 0
        self.state = State.Flat
        
        
        
    def on_tick(self, tick: Tick) -> Signal | None:
        
        if self._update_tick(tick):
            
            self.curr_interval += 1
            # This means that both prices are avaliable so we are happy here
            self._kalman_update()

            if len(self.spread_buffer) >= self.min_buffer and self.curr_interval == self.refit_interval:
                self._fit_ou()
                self._compute_bertram
                self.curr_interval = 0
                
        # We need to be able to send out signals even if we didn't just refit the functions (I think)
        return self._evaluate_signal()
            
        
         
    def _update_tick(self, tick: Tick):
        
        # There are some edge cases which are worth checking sure however most of the 
        # conditions for data should be checked in the base data passer. This is already
        # going to be a busy class so things like checking if tick.price > 0 and whether 
        # it is in the tickers I don't really agree with because it is a waste of time and redundant code
        # the only check I can see being valid is the one to check whether the stock is in the 
        # tickers however if we are going to do this we should make that a set for time complexity
        
        # First task here is to update our dictionary such that the price is current
        self.latest_prices[tick.asset] = tick.price
        
        self.latest_timestamp = tick.timestamp
        self.fresh.append(tick.asset)
        
        if len(self.fresh) == len(self.tickers): return True   
        else: return False
     
    def _kalman_update(self):
       
        x = self.asset_a
        y = self.asset_b
         
        # Taking the initial prediction for the value of beta
        beta_pred = self.beta
        
        # Based on some noise assumption we will take the prediction for the noise of the system
        P_pred = self.P + self.delta
        
        # We now need some dynamic weighting for how we will update the beta and trust the new data
        K = (P_pred * x) / ((P_pred * x**2) + self.r)
        
        # This is our effective spread
        error = y - (beta_pred * x)
        
        self.beta = beta_pred + (K * error)
        self.P = P_pred * (1 - (K * x))
        
        self.spread_buffer.append(error)
        
        # Clear out the list is this right?
        self.fresh = set[str]
      
    def _fit_ou(self):
        
        # Generating the time delay
        x = self.spread_buffer[:-1]
        y = self.spread_buffer[1:]
        
        # returning the line co-efficients
        [a, b] = np.polyfit(x, y, 1)
        
        # the time scale on this may need to be updated
        self.theta = - np.log(a) / self.dt
        
        self.mu = b / (1 - a) if (1-a) > 1e-6 else 0.0
        
        residules = y - (a * x + b)
        
        std_resid = np.std(residules)
        
        # I think I am assuming delta t = 1
        self.sigma = std_resid * np.sqrt( (2 * self.theta) / (1 - a**2) )
       
    def _compute_bertram(self):

        
        # ── Step 1: Transform transaction cost to dimensionless space ──
        # The stationary std of the OU process is sigma / sqrt(2*theta)
        # Divide the real-world cost by this scale factor to get dimensionless cost
        sd = self.sigma / np.sqrt(2 * self.theta)
        dimless_cost = self.cost / sd
        
        def expected_time(a, b):
            
            result, error = quad(lambda y: (np.sqrt(np.pi)/2) *np.exp(y**2) * (1 + erf(y)), a, b)
            
            return result, error
        
        def G(d):
            
            exp, _ = expected_time(-d, d)
            
            return_per_time = (2*d - dimless_cost) / (2 * exp)
            
            return return_per_time
        
        # Optimise the G
        res = minimize_scalar(lambda d: -G(d), bounds = [dimless_cost/2, 4], method='bounded')
        
        world_units = res.x * sd
        
        self.long_threshold = self.mu - world_units
        self.short_threshold = self.mu + world_units
               
    def _evaluate_signal(self) -> Signal | None:
            
        return None