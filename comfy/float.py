import torch

#Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    sign = torch.sign(x)
    abs_x = x.abs()

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)).to(torch.int32) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    mantissa = abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0
    mantissa_scaled = mantissa * (2**MANTISSA_BITS)
    mantissa_floor = mantissa_scaled.floor()
    mantissa = torch.where(
        torch.rand_like(mantissa_scaled) < (mantissa_scaled - mantissa_floor),
        (mantissa_floor + 1) / (2**MANTISSA_BITS),
        mantissa_floor / (2**MANTISSA_BITS)
    )

    # Combine final result calculation
    result = sign * (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + mantissa)

    # Handle zero case
    result = torch.where(abs_x == 0, torch.zeros_like(result), result)

    return result.to(dtype=dtype)



def stochastic_rounding(value, dtype):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        return manual_stochastic_round_to_float8(value, dtype)

    return value.to(dtype=dtype)
