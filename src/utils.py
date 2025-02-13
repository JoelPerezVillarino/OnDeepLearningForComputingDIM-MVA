import time


def timer(func):
    def wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        result = func(*args,**kwargs)
        end_time = time.perf_counter()
        elpased_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elpased_time:.6f} seconds.")
        return result
    return wrapper