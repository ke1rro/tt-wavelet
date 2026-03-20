from __future__ import annotations

from .lifting_runtime import ShiftedArray


class LiftingStep:
    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd


class LiftingStepPredict(LiftingStep):
    def __init__(self, kernel: ShiftedArray):
        self.kernel = kernel

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        pred = even * self.kernel
        return even, odd + pred

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        pred = even * self.kernel
        return even, odd - pred


class LiftingStepUpdate(LiftingStep):
    def __init__(self, kernel: ShiftedArray):
        self.kernel = kernel

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        upd = odd * self.kernel
        return even + upd, odd

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        upd = odd * self.kernel
        return even - upd, odd


class LiftingStepScaleEven(LiftingStep):
    def __init__(self, factor: float):
        self.factor = float(factor)

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even * self.factor, odd

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even * (1.0 / self.factor), odd


class LiftingStepScaleOdd(LiftingStep):
    def __init__(self, factor: float):
        self.factor = float(factor)

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd * self.factor

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd * (1.0 / self.factor)


class LiftingStepSwap(LiftingStep):
    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return odd, even

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return odd, even
