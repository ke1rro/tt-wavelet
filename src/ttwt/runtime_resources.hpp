// SPDX-FileCopyrightText: © 2026 Nikita Lenyk
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <string>

namespace ttwt::detail::runtime_resources {
// Private runtime wiring; no installed C++ API.
void set_root(const std::string &root);
std::string get_root();
std::filesystem::path root_for_use();
std::string kernel_path(const char *relative_path);
std::string scheme_header(const char *quoted_relative_path);
} // namespace ttwt::detail::runtime_resources
