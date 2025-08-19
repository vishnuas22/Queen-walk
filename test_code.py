#!/usr/bin/env python3
"""
Advanced Calculator Module
A comprehensive calculator with multiple mathematical operations
"""

import math
import logging
from typing import Union, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCalculator:
    """
    A sophisticated calculator class with advanced mathematical operations
    """
    
    def __init__(self):
        """Initialize the calculator with default settings"""
        self.history = []
        self.precision = 10
        logger.info("Calculator initialized")
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self._log_operation("add", a, b, result)
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        result = a - b
        self._log_operation("subtract", a, b, result)
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        result = a * b
        self._log_operation("multiply", a, b, result)
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self._log_operation("divide", a, b, result)
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to the power of exponent"""
        result = math.pow(base, exponent)
        self._log_operation("power", base, exponent, result)
        return result
    
    def square_root(self, number: float) -> float:
        """Calculate square root of a number"""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(number)
        self._log_operation("sqrt", number, None, result)
        return result
    
    def factorial(self, n: int) -> int:
        """Calculate factorial of n"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        result = math.factorial(n)
        self._log_operation("factorial", n, None, result)
        return result
    
    def _log_operation(self, operation: str, a: float, b: Optional[float], result: float):
        """Log the operation to history"""
        entry = {
            "operation": operation,
            "operand_a": a,
            "operand_b": b,
            "result": result
        }
        self.history.append(entry)
        logger.info(f"Operation: {operation}({a}, {b}) = {result}")
    
    def get_history(self) -> List[dict]:
        """Get calculation history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history"""
        self.history.clear()
        logger.info("History cleared")

def main():
    """Main function to demonstrate calculator usage"""
    calc = AdvancedCalculator()
    
    # Perform some calculations
    print(f"Addition: 10 + 5 = {calc.add(10, 5)}")
    print(f"Subtraction: 10 - 3 = {calc.subtract(10, 3)}")
    print(f"Multiplication: 4 * 7 = {calc.multiply(4, 7)}")
    print(f"Division: 15 / 3 = {calc.divide(15, 3)}")
    print(f"Power: 2^8 = {calc.power(2, 8)}")
    print(f"Square root: √16 = {calc.square_root(16)}")
    print(f"Factorial: 5! = {calc.factorial(5)}")
    
    # Display history
    print("\nCalculation History:")
    for entry in calc.get_history():
        print(f"  {entry}")

if __name__ == "__main__":
    main()
