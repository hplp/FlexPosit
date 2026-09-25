"""FlexPosit: mixed-precision Posit quantization for LLMs, with matching hardware.

    import flexposit
    model, tok = flexposit.load_model("llama-2-7b")
    state = flexposit.quantize(model, flexposit.FlexPositConfig(bits=4.5),
                               sensitivity="llama-2-7b")
    print(flexposit.wikitext2_perplexity(model, tok))
"""

__version__ = "0.2.0"

from flexposit.api import (FlexPositConfig, QuantState, load_state, quantize, save, sensitivity_csv,
                           shipped_sensitivity)
from flexposit.eval import add_fp8_activation_quant, wikitext2_perplexity
from flexposit.formats import float_quantize, posit_quantize
from flexposit.utils import load_model

__all__ = [
    "FlexPositConfig", "QuantState", "quantize", "save", "load_state", "sensitivity_csv", "shipped_sensitivity",
    "load_model", "wikitext2_perplexity", "add_fp8_activation_quant",
    "posit_quantize", "float_quantize",
]
