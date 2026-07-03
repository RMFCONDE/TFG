function factorial(n::Integer)
    n < 0 && throw(ArgumentError("factorial is not defined for negative integers"))
    return n <= 1 ? 1 : n * factorial(n - 1)
end
