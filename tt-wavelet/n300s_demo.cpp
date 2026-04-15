// TT-Metal N300s Demo with Tracy Profiling
// Simple buffer write/read test on Tenstorrent N300s

#include <iostream>
#include <vector>
#include <cmath>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tt_metal.hpp"
#include <tracy/Tracy.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t N = 1024 * 16;  // 16K elements

int main(int argc, char** argv) {
    ZoneScoped;
    TracyMessageC("N300s Tracy Demo Started", 32, 0x00FFFF);
    
    // Wait for Tracy profiler to connect
    std::cout << "Waiting for Tracy profiler to connect..." << std::endl;
    std::cout << "Connect Tracy GUI to this program, then press Enter to continue..." << std::endl;
    std::cin.get();
    
    IDevice* device = nullptr;
    std::shared_ptr<Buffer> test_buffer;
    
    try {
        // Initialize Device
        {
            ZoneScopedN("Device Initialization");
            TracyMessageC("Initializing N300s device 0...", 32, 0x00FFFF);
            
            device = CreateDevice(0);  // Device ID 0
            
            TracyMessageC("Device initialized successfully", 32, 0x00FF00);
            std::cout << "✓ Device initialized successfully" << std::endl;
        }
        
        // Generate test data
        std::vector<uint32_t> host_data(N);
        {
            ZoneScopedN("Generate Test Data");
            for (uint32_t i = 0; i < N; ++i) {
                host_data[i] = i;
            }
            TracyPlot("Data Size (KB)", (double)(N * sizeof(uint32_t)) / 1024.0);
            std::cout << "✓ Generated " << N << " test values" << std::endl;
        }
        
        // Create buffer
        const size_t buffer_size = N * sizeof(uint32_t);
        {
            ZoneScopedN("Allocate Device Buffer");
            
            test_buffer = CreateBuffer(InterleavedBufferConfig{
                device,
                buffer_size,
                buffer_size,
                BufferType::DRAM
            });
            
            TracyPlot("Device Memory (KB)", (double)buffer_size / 1024.0);
            std::cout << "✓ Allocated DRAM buffer (" << buffer_size << " bytes)" << std::endl;
        }
        
        // Write to device
        {
            ZoneScopedN("Host to Device Transfer");
            
            detail::WriteToBuffer(test_buffer, host_data);
            
            TracyMessageC("Data written to device", 32, 0x00FF00);
            std::cout << "✓ Transferred data to device" << std::endl;
        }
        
        // Read back from device
        std::vector<uint32_t> device_data(N);
        {
            ZoneScopedN("Device to Host Transfer");
            
            detail::ReadFromBuffer(test_buffer, device_data);
            
            TracyMessageC("Data read from device", 32, 0x00FF00);
            std::cout << "✓ Read data back from device" << std::endl;
        }
        
        // Verify
        {
            ZoneScopedN("Verify Results");
            
            uint32_t errors = 0;
            for (uint32_t i = 0; i < N; ++i) {
                if (host_data[i] != device_data[i]) {
                    errors++;
                }
            }
            
            TracyPlot("Verification Errors", (double)errors);
            
            if (errors == 0) {
                TracyMessageC("✓ Verification PASSED - All data matches!", 32, 0x00FF00);
                std::cout << "✓ Verification PASSED - All data matches!" << std::endl;
            } else {
                TracyMessageC("✗ Verification FAILED", 32, 0xFF0000);
                std::cerr << "✗ Verification FAILED: " << errors << " errors" << std::endl;
                return 1;
            }
        }
        
        // Cleanup
        {
            ZoneScopedN("Cleanup");
            test_buffer.reset();
            CloseDevice(device);
            TracyMessageC("N300s Demo Completed Successfully", 32, 0xFF00FF);
            std::cout << "✓ Demo completed successfully" << std::endl;
        }
        
    } catch (const std::exception& e) {
        TracyMessageC("Error occurred", 32, 0xFF0000);
        std::cerr << "Error: " << e.what() << std::endl;
        if (device) {
            CloseDevice(device);
        }
        return 1;
    }
    
    FrameMark;
    return 0;
}
