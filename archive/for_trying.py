from typing import Iterator

def countfrom(n: int) -> Iterator[int]:
    while True:
        yield n
        n += 1

# Example use: printing out the integers from 10 to 20.
# Note that this iteration terminates normally, despite
# countfrom() being written as an infinite loop.

for i in countfrom(10):
    if i <= 20:
        print(i)
    else:
        break

# Another generator, which produces prime numbers indefinitely as needed.