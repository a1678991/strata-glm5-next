// include/strata/program/glm_serve.hpp - `strata --serve` for GLM-5.3-Flash (glm5-next): see src/program/glm_serve.cpp.
// glm_generate.cpp loads the model, the tiers, the session and the prompt path as for a run, then hands them here.
#pragma once

#include "strata/program/glm_generate.hpp"

#include <string>
#include <vector>

namespace strata::core {
struct ModelGeometry;
class GlmEngine;
class GlmExperts;
class FileExpertSource;
class ExpertCache;
}  // namespace strata::core
namespace strata::prefill {
class GlmPrefill;
}

namespace strata::program {

struct GlmServeParts {
    const GlmRunArgs* args = nullptr;
    const core::ModelGeometry* g = nullptr;
    std::vector<std::string> shards;           ///< the GGUF's shards
    core::GlmEngine* eng = nullptr;
    prefill::GlmPrefill* prefill = nullptr;    ///< null: every prompt token through the windows
    core::GlmExperts* experts = nullptr;       ///< null while no MoE layer runs
    core::FileExpertSource* src = nullptr;
    core::ExpertCache* cache = nullptr;        ///< null: no VRAM expert tier
    int window = 1;                            ///< teacher-forced read window (tokens); decode runs windows of 1
    int pool_workers = 0;                      ///< the CPU pool's threads (0: no pool)
};

/// The persistent engine: READY, then GEN requests until QUIT or the end of stdin.  The process exit code.
int glm_serve(const GlmServeParts& parts);

}  // namespace strata::program
