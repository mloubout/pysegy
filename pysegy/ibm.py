"""
Conversion helpers between IBM and IEEE floating point formats.
"""

from typing import Union
import numpy as np


def ibm_to_ieee(value: Union[bytes, bytearray, int]) -> float:
    """
    Convert a 4-byte IBM floating point number to ``float``.

    Parameters
    ----------
    value : bytes or int
        Input value encoded in the IBM 32-bit floating point format.

    Returns
    -------
    float
        Floating point number in IEEE representation.
    """
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 4:
            raise ValueError("IBM float must be 4 bytes")
        val = int.from_bytes(value, byteorder='big', signed=False)
    else:
        val = value & 0xffffffff
    if val == 0:
        return 0.0
    sign = 1 if (val >> 31) == 0 else -1
    exponent = (val >> 24) & 0x7f
    fraction = val & 0x00ffffff
    # IBM exponent is base 16 biased by 64
    mant = fraction / float(0x01000000)
    return sign * mant * 16 ** (exponent - 64)


def ibm_words_to_ieee(vals: np.ndarray) -> np.ndarray:
    """Convert an array of IBM floating-point words to IEEE floats.

    ``vals`` may be strided.  This is useful for decoding samples interleaved
    with SEGY trace headers without first copying every trace into a separate
    bytes object.
    """
    sign = np.where(vals >> 31 == 0, 1.0, -1.0)
    exponent = (vals >> 24) & 0x7F
    fraction = vals & 0x00FFFFFF
    mant = fraction.astype(np.float64) / float(0x01000000)
    out = sign * mant * np.power(16.0, exponent.astype(np.float64) - 64)
    out[vals == 0] = 0.0
    return out.astype(np.float32)


def ibm_to_ieee_array(buf: bytes, count: int, bigendian: bool = True) -> np.ndarray:
    """Vectorized conversion of ``count`` IBM floats contained in ``buf``."""
    dtype = ">u4" if bigendian else "<u4"
    vals = np.frombuffer(buf, dtype=dtype, count=count)
    return ibm_words_to_ieee(vals)


def ieee_to_ibm(f: float) -> bytes:
    """
    Convert ``float`` to IBM 32-bit floating point bytes.

    Parameters
    ----------
    f : float
        Numeric value in IEEE representation.

    Returns
    -------
    bytes
        The value encoded using the IBM format.
    """
    if f == 0.0:
        return b"\x00\x00\x00\x00"
    sign = 0
    if f < 0:
        sign = 0x80
        f = -f
    exponent = 64
    while f < 1.0:
        f *= 16.0
        exponent -= 1
    while f >= 16.0:
        f /= 16.0
        exponent += 1
    fraction = int(f * 0x01000000) & 0x00ffffff
    val = (sign << 24) | (exponent << 24) | fraction
    return val.to_bytes(4, byteorder='big')
