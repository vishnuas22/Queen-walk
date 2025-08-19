#!/usr/bin/env python3
"""
Gamification Services Test Runner

Convenient script to run all gamification tests with proper configuration.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def setup_test_environment():
    """Set up the test environment"""
    # Add the backend directory to Python path
    backend_dir = Path(__file__).parent
    sys.path.insert(0, str(backend_dir))
    
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'INFO'


def run_gamification_tests(test_args=None):
    """Run gamification tests with pytest"""
    setup_test_environment()
    
    # Base pytest command
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_gamification_services.py',
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--asyncio-mode=auto',  # Auto-detect async tests
        '--durations=10',  # Show 10 slowest tests
        '--color=yes',  # Colored output
    ]
    
    # Add additional arguments if provided
    if test_args:
        cmd.extend(test_args)
    
    # Try to add coverage if available
    try:
        import pytest_cov
        cmd.extend([
            '--cov=quantum_intelligence.services.gamification',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov'
        ])
        print("✓ Coverage reporting enabled")
    except ImportError:
        print("ℹ Coverage reporting not available (install pytest-cov for coverage)")
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_test_class(class_name):
    """Run tests for a specific test class"""
    test_args = ['-k', class_name]
    return run_gamification_tests(test_args)


def run_specific_test_method(method_name):
    """Run a specific test method"""
    test_args = ['-k', method_name]
    return run_gamification_tests(test_args)


def run_performance_tests():
    """Run only performance tests"""
    test_args = ['-k', 'Performance', '-s']  # -s to see print output
    return run_gamification_tests(test_args)


def run_integration_tests():
    """Run only integration tests"""
    test_args = ['-k', 'Integration or Scenario', '-s']
    return run_gamification_tests(test_args)


def run_unit_tests():
    """Run only unit tests (exclude integration and performance)"""
    test_args = ['-k', 'not (Integration or Performance or Scenario)']
    return run_gamification_tests(test_args)


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Run gamification services tests')
    parser.add_argument(
        '--class', dest='test_class',
        help='Run tests for a specific class (e.g., TestRewardSystems)'
    )
    parser.add_argument(
        '--method', dest='test_method',
        help='Run a specific test method (e.g., test_reward_optimization)'
    )
    parser.add_argument(
        '--performance', action='store_true',
        help='Run only performance tests'
    )
    parser.add_argument(
        '--integration', action='store_true',
        help='Run only integration tests'
    )
    parser.add_argument(
        '--unit', action='store_true',
        help='Run only unit tests'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Run tests with minimal output for quick feedback'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run tests with debug output'
    )
    parser.add_argument(
        'pytest_args', nargs='*',
        help='Additional arguments to pass to pytest'
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.test_class:
        print(f"Running tests for class: {args.test_class}")
        return run_specific_test_class(args.test_class)
    
    elif args.test_method:
        print(f"Running specific test method: {args.test_method}")
        return run_specific_test_method(args.test_method)
    
    elif args.performance:
        print("Running performance tests...")
        return run_performance_tests()
    
    elif args.integration:
        print("Running integration tests...")
        return run_integration_tests()
    
    elif args.unit:
        print("Running unit tests...")
        return run_unit_tests()
    
    else:
        # Run all tests
        test_args = args.pytest_args or []
        
        if args.fast:
            test_args.extend(['-q', '--tb=line'])
        
        if args.debug:
            test_args.extend(['-s', '--log-cli-level=DEBUG'])
        
        print("Running all gamification tests...")
        return run_gamification_tests(test_args)


if __name__ == '__main__':
    exit_code = main()
    
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("🎉 All tests passed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Some tests failed. Check the output above for details.")
        print("=" * 80)
    
    sys.exit(exit_code)
