# VLA Platform Remote Module
from .rest_client import RESTClient

try:
    from .grpc_client import GRPCClient, GRPC_AVAILABLE
except ImportError:
    GRPCClient = None
    GRPC_AVAILABLE = False

__all__ = [
    "RESTClient",
    "GRPCClient",
    "GRPC_AVAILABLE",
]
