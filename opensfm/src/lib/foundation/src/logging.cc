#include <foundation/logging.h>

#include <iostream>

// pybind11 headers are available because foundation includes
// PYBIND11_INCLUDE_DIR. We use only the C++ API of pybind11 (no module
// registration).
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace foundation {
namespace {

// Obtain (or create) the cached Python logger for "opensfm.<module>".
// The logger objects are cached in a static local map per module name.
// Must be called with the GIL held.
py::object GetLogger(const std::string& module) {
  static std::unordered_map<std::string, py::object> cache;
  auto it = cache.find(module);
  if (it == cache.end()) {
    py::object logger =
        py::module_::import("logging").attr("getLogger")("opensfm." + module);
    cache[module] = logger;
    return logger;
  }
  return it->second;
}

void DoLog(const char* level, const std::string& module,
           const std::string& msg) {
  try {
    py::gil_scoped_acquire gil;
    GetLogger(module).attr(level)(msg);
  } catch (...) {
    std::cerr << "[" << module << "] " << msg << "\n";
  }
}

}  // namespace

void LogInfo(const std::string& module, const std::string& msg) {
  DoLog("info", module, msg);
}

void LogWarning(const std::string& module, const std::string& msg) {
  DoLog("warning", module, msg);
}

void LogDebug(const std::string& module, const std::string& msg) {
  DoLog("debug", module, msg);
}

}  // namespace foundation
