#include <iostream>

#include "tt-metalium/host_api.hpp"

int main() {
    std::cout << "library linked" << std::endl;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    return 0;
}