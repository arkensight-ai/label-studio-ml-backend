from functools import partial, reduce


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def rgetattr(obj: object, attr_str: str) -> object:
    """
    Recursively gets an attribute from a nested object.
    
    Args:
        obj: The root object.
        attr_str: A dot-separated string representing the path to the attribute.
                  e.g., "module.encoder.layer1"
                  
    Returns:
        The target attribute.
    """
    return reduce(getattr, [obj] + attr_str.split('.'))

def rsetattr(obj: object, attr_str: str, value: object):
    """
    Recursively sets an attribute on a nested object.
    
    Args:
        obj: The root object.
        attr_str: A dot-separated string representing the path to the attribute.
                  e.g., "module.encoder.layer1"
        value: The value to set the attribute to.
    """
    pre, _, post = attr_str.rpartition('.')
    parent_obj = rgetattr(obj, pre) if pre else obj
    setattr(parent_obj, post, value)
