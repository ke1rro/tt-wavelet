The computation should be only in fp32 mode.

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[kLwtSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kLwtSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kLwtBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;

This is to avoid path through Src Registers which is used by FPU and copy truncates them to tf32.


The goal of the project is to outperform the PyWavelets library.


To reference isa documentation use [isa docs](../tt-isa-documentation)