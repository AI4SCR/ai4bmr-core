import functools
import logging

# TODO: Implement the enable_verbose decorator
# USECASE: This decorator should be used to enable verbose logging for a function
# def enable_verbose(func):
#     @functools.wraps(func)
#     def function_with_verbose(*args, verbose=1, **kwargs):
#         level = logging.getLogger().level
#         set_stdout_stream_level(verbose)
#         result = func(*args, **kwargs)
#         set_stdout_stream_level(level)
#         return result
#
#     return function_with_verbose
