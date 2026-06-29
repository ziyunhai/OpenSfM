#pragma once

#include <string>

namespace foundation {

// Log to Python's logging module under "opensfm.<module>".
// Falls back to std::cerr if the GIL cannot be acquired (e.g. unit tests).
//
// Usage:
//   foundation::LogInfo("dense", "Fused " + std::to_string(n) + " voxels");
//   foundation::LogWarning("dense", "Hash table overflow");
//   foundation::LogDebug("dense", "iter 0 grad stats ...");
void LogInfo(const std::string& module, const std::string& msg);
void LogWarning(const std::string& module, const std::string& msg);
void LogDebug(const std::string& module, const std::string& msg);

}  // namespace foundation
