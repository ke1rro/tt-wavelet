// SPDX-License-Identifier: Apache-2.0
#include "ttwt/runtime_resources.hpp"
#include "ttwt/runtime_resource_manifest.hpp"

#include <mutex>
#include <stdexcept>

namespace ttwt::detail::runtime_resources {
namespace {
std::mutex resource_mutex;
std::filesystem::path resource_root;
bool resources_used = false;

void require_file(const std::filesystem::path &path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) ||
      std::filesystem::file_size(path, error) == 0 || error) {
    throw std::runtime_error("TT-Wavelet runtime resource is missing or empty: " + path.string());
  }
}

std::filesystem::path validate_root(const std::string &input) {
  std::error_code error;
  auto root = std::filesystem::canonical(input, error);
  if (error || !std::filesystem::is_directory(root)) {
    throw std::runtime_error("TT-Wavelet expected a runtime resource directory: " + input);
  }
  for (const auto relative : required_files) {
    require_file(root / relative);
  }
  return root;
}

void ensure_initialized() {
  if (resource_root.empty()) {
#ifdef TT_WAVELET_DEVELOPMENT_RESOURCE_ROOT
    resource_root = validate_root(TT_WAVELET_DEVELOPMENT_RESOURCE_ROOT);
#else
    throw std::runtime_error("TT-Wavelet runtime resource root is not initialized; import ttwt to "
                             "configure package resources");
#endif
  }
}
} // namespace

void set_root(const std::string &input) {
  auto root = validate_root(input);
  std::lock_guard lock(resource_mutex);
  if (resources_used && root != resource_root) {
    throw std::runtime_error(
        "TT-Wavelet cannot change the runtime resource root after active use: " +
        resource_root.string());
  }
  resource_root = std::move(root);
}

std::string get_root() {
  std::lock_guard lock(resource_mutex);
  ensure_initialized();
  return resource_root.string();
}

std::filesystem::path root_for_use() {
  std::lock_guard lock(resource_mutex);
  ensure_initialized();
  resources_used = true;
  return resource_root;
}

std::string kernel_path(const char *relative_path) {
  auto path = root_for_use() / "ttwt/device/kernels" / relative_path;
  require_file(path);
  return path.string();
}

std::string scheme_header(const char *quoted_relative_path) {
  const std::string quoted(quoted_relative_path);
  auto path = root_for_use() / quoted.substr(1, quoted.size() - 2);
  require_file(path);
  return "\"" + path.string() + "\"";
}
} // namespace ttwt::detail::runtime_resources
