# Test Python file for multi-modal AI processing
# This file demonstrates various Python concepts

def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    """A simple data processing class"""
    
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        """Process the data"""
        if isinstance(self.data, list):
            self.data = [x * 2 for x in self.data if isinstance(x, (int, float))]
        self.processed = True
        return self.data
    
    def get_stats(self):
        """Get basic statistics"""
        if not self.processed:
            self.process()
        
        if self.data:
            return {
                'count': len(self.data),
                'sum': sum(self.data),
                'average': sum(self.data) / len(self.data),
                'max': max(self.data),
                'min': min(self.data)
            }
        return {}

# Example usage
if __name__ == "__main__":
    # Test Fibonacci
    print("Fibonacci sequence:")
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
    
    # Test DataProcessor
    processor = DataProcessor([1, 2, 3, 4, 5])
    result = processor.process()
    stats = processor.get_stats()
    
    print(f"\nProcessed data: {result}")
    print(f"Statistics: {stats}")
