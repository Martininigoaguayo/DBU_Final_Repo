from typing import Callable, Any
from prefect import task
from datetime import timedelta

def task_wrapper(func=None, *, use_cache=False, cache_for=timedelta(minutes=10), **kwargs):
    """
    A decorator function to wrap any arbitrary function with the @task decorator. This allows the wrapped function
    to communicate with a local Prefect server, enabling it to run as a Prefect task in workflows.
    
    Caching can be optionally enabled by passing use_cache=True.

    Args:
        func (callable): The function to be wrapped and executed as a Prefect task.
        use_cache (bool): If True, enables caching for the task.
        cache_for (timedelta): The duration to cache the result, if caching is enabled.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        callable: The wrapped Prefect task.
    """
    
    # Cache key function (optional, only used when caching is enabled)
    def default_cache_key_fn(task, parameters):
    
        return str(hash(parameters.values()))

    # Create a task with or without caching depending on use_cache
    if use_cache:
        return task(cache_expiration=cache_for, cache_key_fn=default_cache_key_fn)(func)
    else:
        return task(func)



