from typing import List

def factorial(n: int) -> int:
    """Calculate the factorial of a number (recursive)."""
    if n < 0:
        raise ValueError("factorial is not defined for negative integers")
    return 1 if n <= 1 else n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number (recursive)."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def gcd(a: int, b: int) -> int:
    """Greatest common divisor using Euclid's algorithm."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def primes_up_to(limit: int) -> List[int]:
    """Return a list of all prime numbers up to limit (inclusive)."""
    if limit < 2:
        return []
    primes: List[int] = []
    for i in range(2, limit + 1):
        is_prime = True
        for p in primes:
            if i % p == 0:
                is_prime = False
                break
            if p * p > i:
                break
        if is_prime:
            primes.append(i)
    return primes

def example_function(x: int):
    """Example function to mirror the Lisp version's behavior."""
    result = [factorial(x), fibonacci(x), gcd(x, 10), primes_up_to(x)]
    print(f"Factorial of {x}: {result[0]}")
    print(f"Fibonacci of {x}: {result[1]}")
    print(f"GCD of {x} and 10: {result[2]}")
    print(f"Primes up to {x}: {result[3]}")
    return result


if __name__ == '__main__':
    example_function(10)
