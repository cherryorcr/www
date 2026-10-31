"""Change the frozen solver's explicit cast, without changing its mathematics."""
import ast
import inspect


def install(module):
    original = module._log_sinkhorn
    tree = ast.parse(inspect.getsource(original))
    changed = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'float' and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'scores'):
            node.func.attr = 'double'
            changed += 1
    if changed != 1:
        raise AssertionError('Unexpected frozen solver cast; review source before changing precision')
    namespace = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(tree), '<cotag-sinkhorn-float64>', 'exec'), namespace)
    native_double = namespace['_log_sinkhorn']
    module._cotag_precision_calls = 0

    def precise(scores, column_mass, *args, **kwargs):
        result = native_double(scores.double(), column_mass.double(), *args, **kwargs)
        if result.dtype != module.torch.float64:
            raise AssertionError('The solver did not actually retain float64 arithmetic')
        module._cotag_precision_calls += 1
        return result.to(scores.dtype)

    module._log_sinkhorn = precise
    return original
