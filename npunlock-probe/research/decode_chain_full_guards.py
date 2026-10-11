"""Save the model guards alongside the full-stack default-routing checks."""
import decode_chain_api as common
from decode_chain_full import OUT
import decode_chain_regressions as regression

common.OUT=OUT
regression.OUT=OUT
if __name__=='__main__':regression.guards()
