# Based on https://github.com/ke1rro/tt-wavelet/blob/b17988c7b25192cd7f96c13d475c24ccf52344cb/experiments/precision/dtypes.py
import numba as nb
import numpy as np
from numba import types
from numba.extending import intrinsic


@intrinsic
def bitcast_u32_f32(typingctx, x):
    if x != types.uint32:
        return None

    sig = types.float32(types.uint32)

    def codegen(context, builder, signature, args):
        f32 = context.get_value_type(types.float32)
        return builder.bitcast(args[0], f32)

    return sig, codegen


@intrinsic
def bitcast_f32_u32(typingctx, x):
    if x != types.float32:
        return None

    sig = types.uint32(types.float32)

    def codegen(context, builder, signature, args):
        u32 = context.get_value_type(types.uint32)
        return builder.bitcast(args[0], u32)

    return sig, codegen


@nb.njit(nb.float32(nb.float32), inline="always")
def bf16_round(x):
    xf = np.float32(x)
    u = bitcast_f32_u32(xf)

    lsb = (u >> np.uint32(16)) & np.uint32(1)
    u = (u + (np.uint32(0x7FFF) + lsb)) & np.uint32(0xFFFF0000)

    return bitcast_u32_f32(np.uint32(u))


@nb.njit(nb.float32(nb.float32), inline="always")
def tf32_round(x):
    xf = np.float32(x)
    u = bitcast_f32_u32(xf)

    lsb = (u >> np.uint32(13)) & np.uint32(1)
    u = (u + (np.uint32(0x1FFF) + lsb)) & np.uint32(0xFFFFE000)

    return bitcast_u32_f32(np.uint32(u))


class dtype_:
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Convolution not implemented for this dtype")

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Addition not implemented for this dtype")

    @staticmethod
    def mul(data: np.ndarray, value: float):
        raise NotImplementedError("Multiplication not implemented for this dtype")


dtype = type[dtype_]


class float32(dtype_):
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=np.float32)
        kernel = np.array(kernel, dtype=np.float32)
        return np.convolve(data, kernel, mode="valid").astype(np.float64)

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        return (data1 + data2).astype(np.float64)

    @staticmethod
    def mul(data: np.ndarray, value: float):
        data = np.array(data, dtype=np.float32)
        return (data * value).astype(np.float64)


class float64(dtype_):
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=np.float64)
        kernel = np.array(kernel, dtype=np.float64)
        return np.convolve(data, kernel, mode="valid")

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1 = np.array(data1, dtype=np.float64)
        data2 = np.array(data2, dtype=np.float64)
        return data1 + data2

    @staticmethod
    def mul(data: np.ndarray, value: float):
        data = np.array(data, dtype=np.float64)
        return (data * value).astype(np.float64)


class bfloat16(dtype_):
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=np.float32)
        kernel = np.array(kernel, dtype=np.float32)
        result = np.zeros(len(data) - len(kernel) + 1, dtype=np.float32)
        bfloat16._conv(data, kernel, result)
        return result.astype(np.float64)

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        result = np.zeros_like(data1, dtype=np.float32)
        bfloat16._add(data1, data2, result)
        return result.astype(np.float64)

    @staticmethod
    def mul(data: np.ndarray, value: float):
        data = np.array(data, dtype=np.float32)
        result = np.zeros_like(data, dtype=np.float32)
        bfloat16._mul(data, np.float32(value), result)
        return result.astype(np.float64)

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32, nb.float32[:]))
    def _mul(data: np.ndarray, value: float, result: np.ndarray):
        for i in range(len(result)):
            result[i] = bf16_round(bf16_round(data[i]) * bf16_round(value))

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _conv(data: np.ndarray, kernel: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            acc = np.float32(0.0)
            for j in range(len(kernel)):
                acc = bf16_round(
                    acc + bf16_round(data[i + j]) * bf16_round(kernel[len(kernel) - 1 - j])
                )
            result[i] = acc

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _add(data1: np.ndarray, data2: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            result[i] = bf16_round(bf16_round(data1[i]) + bf16_round(data2[i]))


class tfloat32(dtype_):
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=np.float32)
        kernel = np.array(kernel, dtype=np.float32)
        result = np.zeros(len(data) - len(kernel) + 1, dtype=np.float32)
        tfloat32._conv(data, kernel, result)
        return result.astype(np.float64)

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        result = np.zeros_like(data1, dtype=np.float32)
        tfloat32._add(data1, data2, result)
        return result.astype(np.float64)

    @staticmethod
    def mul(data: np.ndarray, value: float):
        data = np.array(data, dtype=np.float32)
        result = np.zeros_like(data, dtype=np.float32)
        tfloat32._mul(data, np.float32(value), result)
        return result.astype(np.float64)

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32, nb.float32[:]))
    def _mul(data: np.ndarray, value: float, result: np.ndarray):
        for i in range(len(result)):
            result[i] = tf32_round(tf32_round(data[i]) * tf32_round(value))

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _conv(data: np.ndarray, kernel: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            acc = np.float32(0.0)
            for j in range(len(kernel)):
                acc = tf32_round(
                    acc + tf32_round(data[i + j]) * tf32_round(kernel[len(kernel) - 1 - j])
                )
            result[i] = acc

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _add(data1: np.ndarray, data2: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            result[i] = tf32_round(tf32_round(data1[i]) + tf32_round(data2[i]))


class mixed_bf16xf32(dtype_):
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=np.float32)
        kernel = np.array(kernel, dtype=np.float32)
        result = np.zeros(len(data) - len(kernel) + 1, dtype=np.float32)
        mixed_bf16xf32._conv(data, kernel, result)
        return result.astype(np.float64)

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        result = np.zeros_like(data1, dtype=np.float32)
        mixed_bf16xf32._add(data1, data2, result)
        return result.astype(np.float64)

    @staticmethod
    def mul(data: np.ndarray, value: float):
        data = np.array(data, dtype=np.float32)
        result = np.zeros_like(data, dtype=np.float32)
        mixed_bf16xf32._mul(data, np.float32(value), result)
        return result.astype(np.float64)

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32, nb.float32[:]))
    def _mul(data: np.ndarray, value: float, result: np.ndarray):
        for i in range(len(result)):
            result[i] = bf16_round(data[i]) * bf16_round(value)

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _conv(data: np.ndarray, kernel: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            acc = np.float32(0.0)
            for j in range(len(kernel)):
                acc += bf16_round(data[i + j]) * bf16_round(kernel[len(kernel) - 1 - j])
            result[i] = acc

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _add(data1: np.ndarray, data2: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            result[i] = data1[i] + data2[i]


class mixed_tf32xf32(dtype_):
    @staticmethod
    def conv(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=np.float32)
        kernel = np.array(kernel, dtype=np.float32)
        result = np.zeros(len(data) - len(kernel) + 1, dtype=np.float32)
        mixed_tf32xf32._conv(data, kernel, result)
        return result.astype(np.float64)

    @staticmethod
    def add(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        result = np.zeros_like(data1, dtype=np.float32)
        mixed_tf32xf32._add(data1, data2, result)
        return result.astype(np.float64)

    @staticmethod
    def mul(data: np.ndarray, value: float):
        data = np.array(data, dtype=np.float32)
        result = np.zeros_like(data, dtype=np.float32)
        mixed_tf32xf32._mul(data, np.float32(value), result)
        return result.astype(np.float64)

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32, nb.float32[:]))
    def _mul(data: np.ndarray, value: float, result: np.ndarray):
        for i in range(len(result)):
            result[i] = tf32_round(data[i]) * tf32_round(value)

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _conv(data: np.ndarray, kernel: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            acc = np.float32(0.0)
            for j in range(len(kernel)):
                acc += tf32_round(data[i + j]) * tf32_round(kernel[len(kernel) - 1 - j])
            result[i] = acc

    @staticmethod
    @nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.float32[:]))
    def _add(data1: np.ndarray, data2: np.ndarray, result: np.ndarray):
        for i in range(len(result)):
            result[i] = data1[i] + data2[i]


map = {
    "float64": float64,
    "float32": float32,
    "tfloat32": tfloat32,
    "bfloat16": bfloat16,
    "mixed_bf16xf32": mixed_bf16xf32,
    "mixed_tf32xf32": mixed_tf32xf32,
}
