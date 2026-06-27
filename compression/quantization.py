<<<<<<< HEAD
from .linear import ModifiedLinear
import numpy as np

def quantize_model(model, k=32):
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, ModifiedLinear):
            module.quantize(k)
            layer_count += 1
            print(f"Quantized {name} to {k} clusters.")
    
    print(f"Quantization complete. {layer_count} layers quantized.")
=======
from .linear import ModifiedLinear
import numpy as np

def quantize_model(model, k=32):
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, ModifiedLinear):
            module.quantize(k)
            layer_count += 1
            print(f"Quantized {name} to {k} clusters.")
    
    print(f"Quantization complete. {layer_count} layers quantized.")
>>>>>>> 979e0d06912afe50af22e2bee29e80fcac758fee
    return model