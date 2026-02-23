// Tracy Profiler Demo - Feature Showcase
// Each section demonstrates a different Tracy profiling feature

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <chrono>
#include <cstring>

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

// Global counter for memory tracking demo
static size_t g_total_allocated = 0;

//=============================================================================
// DEMO 1: Basic Zone Profiling - ZoneScoped
//=============================================================================
void demo_basic_zones() {
    ZoneScoped;  // Automatically profiles entire function
    
    std::cout << "\n[1] Basic Zone Profiling" << std::endl;
    
    // Fast operation
    {
        ZoneScopedN("Fast Math");
        double result = 0.0;
        for (int i = 0; i < 1000; ++i) {
            result += std::sin(i * 0.1);
        }
    }
    
    // Slow operation
    {
        ZoneScopedN("Slow Computation");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Nested zones
    {
        ZoneScopedN("Outer Zone");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        {
            ZoneScopedN("Inner Zone 1");
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
        
        {
            ZoneScopedN("Inner Zone 2");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

//=============================================================================
// DEMO 2: Zone Text & Dynamic Names
//=============================================================================
void demo_zone_text(int iteration) {
    ZoneScopedN("Process Item");
    
    // Add dynamic text to zone (shows up in Tracy as zone name suffix)
    char name[64];
    snprintf(name, sizeof(name), "Item #%d", iteration);
    ZoneText(name, strlen(name));
    
    std::cout << "[2] Zone Text: Processing " << name << std::endl;
    
    // Simulate variable work time based on iteration
    int delay_ms = 10 + (iteration % 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
}

//=============================================================================
// DEMO 3: Tracy Messages & Colors
//=============================================================================
void demo_messages() {
    ZoneScoped;
    
    std::cout << "\n[3] Tracy Messages with Colors" << std::endl;
    
    // Simple message
    TracyMessageL("Starting message demo");
    
    // Colored messages (RGB hex colors)
    TracyMessageC("Info message", 12, 0x00FF00);      // Green
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    TracyMessageC("Warning message", 15, 0xFFFF00);   // Yellow
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    TracyMessageC("Error message", 13, 0xFF0000);     // Red
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    TracyMessageC("Debug message", 13, 0x00FFFF);     // Cyan
    
    // Dynamic message
    char msg[128];
    int value = 42;
    snprintf(msg, sizeof(msg), "Computed value: %d", value);
    TracyMessage(msg, strlen(msg));
}

//=============================================================================
// DEMO 4: Plotting Values
//=============================================================================
void demo_plots() {
    ZoneScoped;
    
    std::cout << "\n[4] Tracy Plots" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for (int i = 0; i < 50; ++i) {
        // Multiple plots can be tracked simultaneously
        double cpu_usage = 50.0 + 30.0 * std::sin(i * 0.2);
        double memory_mb = 100.0 + dis(gen);
        double temperature = 60.0 + 10.0 * std::cos(i * 0.15);
        
        TracyPlot("CPU Usage %", cpu_usage);
        TracyPlot("Memory MB", memory_mb);
        TracyPlot("Temperature C", temperature);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

//=============================================================================
// DEMO 5: Memory Tracking
//=============================================================================
void demo_memory_tracking() {
    ZoneScoped;
    
    std::cout << "\n[5] Memory Allocation Tracking" << std::endl;
    
    const int num_allocs = 5;
    void* ptrs[num_allocs];
    size_t sizes[num_allocs] = {1024, 2048, 4096, 8192, 16384};
    
    // Allocate memory blocks
    for (int i = 0; i < num_allocs; ++i) {
        ZoneScopedN("Allocate Block");
        
        ptrs[i] = malloc(sizes[i]);
        TracyAlloc(ptrs[i], sizes[i]);  // Tell Tracy about allocation
        
        g_total_allocated += sizes[i];
        TracyPlot("Total Memory Bytes", (double)g_total_allocated);
        
        char msg[64];
        snprintf(msg, sizeof(msg), "Allocated %zu bytes", sizes[i]);
        TracyMessage(msg, strlen(msg));
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Free memory blocks
    for (int i = 0; i < num_allocs; ++i) {
        ZoneScopedN("Free Block");
        
        TracyFree(ptrs[i]);  // Tell Tracy about deallocation
        free(ptrs[i]);
        
        g_total_allocated -= sizes[i];
        TracyPlot("Total Memory Bytes", (double)g_total_allocated);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

//=============================================================================
// DEMO 6: Frame Marking (for frame-based profiling)
//=============================================================================
void demo_frame_profiling() {
    ZoneScoped;
    
    std::cout << "\n[6] Frame-Based Profiling" << std::endl;
    
    const int num_frames = 30;
    
    for (int frame = 0; frame < num_frames; ++frame) {
        ZoneScopedN("Game Frame");  // Each iteration is a "frame"
        
        // Simulate frame work
        {
            ZoneScopedN("Update Logic");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        {
            ZoneScopedN("Render");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        {
            ZoneScopedN("Audio");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        
        // Track FPS
        double fps = 1000.0 / 17.0;  // ~60 FPS target
        TracyPlot("FPS", fps);
        TracyPlot("Frame Number", (double)frame);
        
        // IMPORTANT: Mark frame boundary!
        // This tells Tracy where one frame ends and another begins
        FrameMark;
    }
}

//=============================================================================
// DEMO 7: Conditional Profiling & Performance Comparison
//=============================================================================
void slow_sort(std::vector<int>& data) {
    ZoneScopedN("Bubble Sort");
    
    // Intentionally slow bubble sort
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data.size() - 1; ++j) {
            if (data[j] > data[j + 1]) {
                std::swap(data[j], data[j + 1]);
            }
        }
    }
}

void fast_sort(std::vector<int>& data) {
    ZoneScopedN("STL Sort");
    std::sort(data.begin(), data.end());
}

void demo_performance_comparison() {
    ZoneScoped;
    
    std::cout << "\n[7] Performance Comparison" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    
    // Test 1: Slow algorithm
    {
        std::vector<int> data(200);
        for (auto& val : data) val = dis(gen);
        
        ZoneScopedN("Test Slow Algorithm");
        slow_sort(data);
        TracyMessageC("Slow sort completed", 19, 0xFF0000);
    }
    
    // Test 2: Fast algorithm
    {
        std::vector<int> data(200);
        for (auto& val : data) val = dis(gen);
        
        ZoneScopedN("Test Fast Algorithm");
        fast_sort(data);
        TracyMessageC("Fast sort completed", 19, 0x00FF00);
    }
}

//=============================================================================
// DEMO 8: Multithreaded Profiling
//=============================================================================
void worker_thread(int thread_id) {
    // Tracy automatically tracks per-thread performance
    ZoneScopedN("Worker Thread");
    
    char name[32];
    snprintf(name, sizeof(name), "Thread %d", thread_id);
    ZoneText(name, strlen(name));
    
    for (int i = 0; i < 5; ++i) {
        ZoneScopedN("Work Item");
        
        char msg[64];
        snprintf(msg, sizeof(msg), "Thread %d processing item %d", thread_id, i);
        TracyMessage(msg, strlen(msg));
        
        // Variable work time
        int delay = 10 + (thread_id * 5) + (i * 3);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
}

void demo_multithreading() {
    ZoneScoped;
    
    std::cout << "\n[8] Multithreaded Profiling" << std::endl;
    
    const int num_threads = 4;
    std::vector<std::thread> threads;
    
    TracyMessageC("Starting worker threads", 23, 0x00FFFF);
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_thread, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    TracyMessageC("All threads completed", 21, 0x00FF00);
}

//=============================================================================
// Main Demo Runner
//=============================================================================
int main() {
    ZoneScopedN("Main");
    
    std::cout << "========================================" << std::endl;
    std::cout << "   Tracy Profiler Feature Showcase" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nConnect Tracy GUI to see profiling data!" << std::endl;
    std::cout << "Run ./tracy-gui and connect to this app." << std::endl;
    
    TracyMessageC("=== Tracy Demo Started ===", 26, 0xFF00FF);
    
    // Run all demos
    demo_basic_zones();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < 3; ++i) {
        demo_zone_text(i);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    demo_messages();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    demo_plots();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    demo_memory_tracking();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    demo_frame_profiling();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    demo_performance_comparison();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    demo_multithreading();
    
    TracyMessageC("=== All Demos Complete ===", 26, 0x00FF00);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo complete! Check Tracy GUI for:" << std::endl;
    std::cout << "  • Zone hierarchies & timing" << std::endl;
    std::cout << "  • Memory allocation graphs" << std::endl;
    std::cout << "  • Custom plots (CPU, Memory, etc.)" << std::endl;
    std::cout << "  • Colored messages & logs" << std::endl;
    std::cout << "  • Frame timing analysis" << std::endl;
    std::cout << "  • Multi-thread visualization" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
