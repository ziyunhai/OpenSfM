#pragma once

// OpenCL kernels for Perona-Malik anisotropic diffusion

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline const char* kDiffusionKernelSource = R"CL(

// ----------------------------------------------------------------
// Perona-Malik anisotropic diffusion — one iteration.
// Fills NaN holes while preserving edges (Dirichlet BC on observed
// cells, zero-flux at data boundary).
//
// 2D dispatch: global_id(0) = x, global_id(1) = y.
//
// guide: edge magnitude from coarser level (or self for coarsest).
//        Used as |∇| in the edge-stopping function.
// valid_mask: 1 where finalize produced a value (Dirichlet — frozen).
// ----------------------------------------------------------------
__kernel void dsm_diffuse(
    __global const float* input,
    __global float* output,
    __global const float* guide,
    __global const unsigned char* valid_mask,
    const int width,
    const int height,
    const float kappa,
    const float dt)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // Observed cells are frozen (Dirichlet boundary condition).
    if (valid_mask[idx]) {
        output[idx] = input[idx];
        return;
    }

    const float center = input[idx];
    // If this cell has never been touched (still NaN at start),
    // check if any neighbor is valid to seed diffusion.
    if (isnan(center)) {
        // Try to initialize from mean of valid neighbors.
        float sum = 0.0f;
        int count = 0;
        if (x > 0 && !isnan(input[idx - 1]))         { sum += input[idx - 1]; ++count; }
        if (x < width - 1 && !isnan(input[idx + 1])) { sum += input[idx + 1]; ++count; }
        if (y > 0 && !isnan(input[idx - width]))      { sum += input[idx - width]; ++count; }
        if (y < height - 1 && !isnan(input[idx + width])) { sum += input[idx + width]; ++count; }
        output[idx] = (count > 0) ? (sum / (float)count) : NAN;
        return;
    }

    // 4-neighbor anisotropic diffusion.
    // Edge-stopping: g(x) = 1 / (1 + (x/kappa)^2)  [Perona-Malik Type 1]
    const float inv_kappa_sq = 1.0f / (kappa * kappa);
    float flux_sum = 0.0f;

    // Helper: conductance at midpoint between cell and neighbor.
    // Use max of guide values at center and neighbor as edge magnitude.
    const float g_center = guide[idx];

    // East
    if (x < width - 1) {
        const float nval = input[idx + 1];
        if (!isnan(nval)) {
            const float g_n = guide[idx + 1];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }
    // West
    if (x > 0) {
        const float nval = input[idx - 1];
        if (!isnan(nval)) {
            const float g_n = guide[idx - 1];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }
    // South
    if (y < height - 1) {
        const float nval = input[idx + width];
        if (!isnan(nval)) {
            const float g_n = guide[idx + width];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }
    // North
    if (y > 0) {
        const float nval = input[idx - width];
        if (!isnan(nval)) {
            const float g_n = guide[idx - width];
            const float g_max = fmax(g_center, g_n);
            const float cond = 1.0f / (1.0f + g_max * g_max * inv_kappa_sq);
            flux_sum += cond * (nval - center);
        }
    }

    output[idx] = center + dt * flux_sum;
}

// ----------------------------------------------------------------
// Compute gradient magnitude (Sobel-like) of a float grid.
// Used to generate the diffusion guide from a DSM level.
// 2D dispatch: global_id(0) = x, global_id(1) = y.
// ----------------------------------------------------------------
__kernel void dsm_gradient_magnitude(
    __global const float* input,
    __global float* output,
    const int width,
    const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const float c = input[idx];
    if (isnan(c)) {
        output[idx] = 0.0f;
        return;
    }

    // Central differences with NaN handling (fall back to one-sided).
    float gx = 0.0f, gy = 0.0f;

    const float left  = (x > 0) ? input[idx - 1] : NAN;
    const float right = (x < width - 1) ? input[idx + 1] : NAN;
    if (!isnan(left) && !isnan(right)) {
        gx = (right - left) * 0.5f;
    } else if (!isnan(right)) {
        gx = right - c;
    } else if (!isnan(left)) {
        gx = c - left;
    }

    const float up   = (y > 0) ? input[idx - width] : NAN;
    const float down = (y < height - 1) ? input[idx + width] : NAN;
    if (!isnan(up) && !isnan(down)) {
        gy = (down - up) * 0.5f;
    } else if (!isnan(down)) {
        gy = down - c;
    } else if (!isnan(up)) {
        gy = c - up;
    }

    output[idx] = sqrt(gx * gx + gy * gy);
})CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
