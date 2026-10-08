// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ttwt::benchmark {
inline std::string read_file(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::invalid_argument("Cannot read " + path.string());
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

inline std::string sha256(const std::string &bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                  EVP_MD_CTX_free);
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1 ||
      EVP_DigestFinal_ex(context.get(), hash, &length) != 1) {
    throw std::runtime_error("SHA-256 calculation failed");
  }
  std::ostringstream result;
  for (unsigned int i = 0; i < length; ++i) {
    result << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(hash[i]);
  }
  return result.str();
}

inline nlohmann::json load_configuration(const std::filesystem::path &toml,
                                         const std::filesystem::path &manifest) {
  auto plan = nlohmann::json::parse(read_file(manifest));
  auto prepared = plan;
  prepared.erase("prepared_sha256");
  if (plan.at("prepared_sha256") != sha256(prepared.dump())) {
    throw std::invalid_argument("Prepared manifest checksum mismatch; regenerate from TOML");
  }
  if (plan.at("schema_version") != 1 || plan.at("config_sha256") != sha256(read_file(toml))) {
    throw std::invalid_argument(
        "Manifest does not match the source TOML; regenerate it with prepare_campaign.py");
  }
  if (plan.at("matrix_sha256") != sha256(plan.at("workloads").dump())) {
    throw std::invalid_argument("Manifest workload hash mismatch");
  }
  if (plan.at("workloads").empty()) {
    throw std::invalid_argument("Empty workload manifest");
  }
  return plan;
}
} // namespace ttwt::benchmark
