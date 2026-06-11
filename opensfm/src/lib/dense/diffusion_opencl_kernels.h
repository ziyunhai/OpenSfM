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
}

// ----------------------------------------------------------------
// Joint (cross) bilateral filter of the DSM guided by the ortho colour.
// Averages each cell's height only with neighbours of SIMILAR ortho colour, so
// the height field smooths WITHIN a roof / within the ground but NOT across the
// ortho's (photo-sharp) roof<->ground colour edge.  Over a few iterations a
// fattened height ramp therefore collapses onto the colour edge -> sharp,
// photo-aligned building outlines.  Invalid (NaN-hole) cells pass through
// unchanged and never contribute as neighbours, so no-data borders are kept.
// guide: RGB in [0,1], 3*ncells.  2D dispatch: id(0)=x, id(1)=y.
// ----------------------------------------------------------------
__kernel void dsm_joint_bilateral(
    __global const float* dsm_in,
    __global float*       dsm_out,
    __global const float* guide,
    __global const unsigned char* valid,
    const int   width,
    const int   height,
    const int   radius,
    const float inv_2s2_spatial,
    const float inv_2s2_range)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int c = y * width + x;

    if (!valid[c]) { dsm_out[c] = dsm_in[c]; return; }  // keep no-data holes

    const float gr = guide[3 * c + 0];
    const float gg = guide[3 * c + 1];
    const float gb = guide[3 * c + 2];

    float sum = 0.0f, wsum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            int n = yy * width + xx;
            if (!valid[n]) continue;
            float dr = guide[3 * n + 0] - gr;
            float dg = guide[3 * n + 1] - gg;
            float db = guide[3 * n + 2] - gb;
            float range = dr * dr + dg * dg + db * db;
            float spatial = (float)(dx * dx + dy * dy);
            float w = exp(-spatial * inv_2s2_spatial - range * inv_2s2_range);
            sum += w * dsm_in[n];
            wsum += w;
        }
    }
    dsm_out[c] = (wsum > 0.0f) ? (sum / wsum) : dsm_in[c];
}

// ----------------------------------------------------------------
// Coherence-enhancing shock filter (Weickert) for the DSM.
// Sharpens a fattened height ramp into a step WITHOUT the ortho, so it breaks
// the bake<->DSM chicken-and-egg.  The shock direction is the second derivative
// along the COHERENT edge normal w = (cosθ,sinθ) from a local structure tensor
// (window 'win'), which steers the sharpening to be straight + consistent, and
// an upwind (Osher-Sethian) dilation/erosion pushes each side of the ramp
// toward its plateau so the transition collapses to one cell.  NaN (no-data)
// cells, the grid border, and cells touching a hole pass through unchanged.
// 2D dispatch: id(0)=x, id(1)=y.
// ----------------------------------------------------------------
__kernel void dsm_shock(
    __global const float* u_in,
    __global float*       u_out,
    __global const unsigned char* valid,
    const int   width,
    const int   height,
    const int   win,
    const float dt,
    const float coherence,
    const float gsd,
    const float edge_slope)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int c = y * width + x;
    if (!valid[c]) { u_out[c] = u_in[c]; return; }

    // Need the 8 neighbours for centred derivatives; bail on border / no-data.
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        u_out[c] = u_in[c]; return;
    }
    const int cL=c-1, cR=c+1, cD=c-width, cU=c+width;
    const int cLD=c-width-1, cRD=c-width+1, cLU=c+width-1, cRU=c+width+1;
    if (!(valid[cL] && valid[cR] && valid[cD] && valid[cU] &&
          valid[cLD] && valid[cRD] && valid[cLU] && valid[cRU])) {
        u_out[c] = u_in[c]; return;
    }

    const float u  = u_in[c];
    const float uL = u_in[cL], uR = u_in[cR], uD = u_in[cD], uU = u_in[cU];
    const float uLD=u_in[cLD], uRD=u_in[cRD], uLU=u_in[cLU], uRU=u_in[cRU];

    // Structure tensor over a (2*win+1)^2 window (orientation/integration scale).
    float j11=0.0f, j12=0.0f, j22=0.0f;
    int cnt=0;
    for (int dy=-win; dy<=win; dy++) {
        int yy=y+dy; if (yy<1 || yy>=height-1) continue;
        for (int dx=-win; dx<=win; dx++) {
            int xx=x+dx; if (xx<1 || xx>=width-1) continue;
            int p=yy*width+xx;
            if (!(valid[p-1] && valid[p+1] &&
                  valid[p-width] && valid[p+width])) continue;
            float gx=0.5f*(u_in[p+1]-u_in[p-1]);
            float gy=0.5f*(u_in[p+width]-u_in[p-width]);
            j11+=gx*gx; j12+=gx*gy; j22+=gy*gy; cnt++;
        }
    }

    // Dominant (across-edge) direction.  atan2 is scale-invariant, so the raw
    // (unnormalised) tensor gives the same orientation.
    float theta = 0.5f*atan2(2.0f*j12, j11-j22);
    float wx = cos(theta), wy = sin(theta);

    // Edge-strength gate: only let the shock fire where the surface is genuinely
    // STEP-like (high slope), so smooth gradients / gentle slopes are NOT
    // terraced into staircases.  grad_rms = RMS height gradient over the window
    // (height units / cell); slope = grad_rms / gsd is the dimensionless
    // rise/run.  Buildings edges have slope >> 1 even when fattened; terrain and
    // pitched roofs are gentle.  smoothstep ramps the shock 0->1 across the
    // threshold so there is no hard sharpening boundary.
    float grad_rms = (cnt > 0) ? sqrt((j11 + j22) / (float)cnt) : 0.0f;
    float slope = grad_rms / fmax(gsd, 1e-6f);
    float w_edge = smoothstep(0.5f * edge_slope, edge_slope, slope);

    // Second directional derivative along w.
    float uxx = uR - 2.0f*u + uL;
    float uyy = uU - 2.0f*u + uD;
    float uxy = 0.25f*(uRU - uRD - uLU + uLD);
    float uww = wx*wx*uxx + 2.0f*wx*wy*uxy + wy*wy*uyy;
    // Tangential (along-edge) second derivative, v = (-wy, wx).
    float uvv = wy*wy*uxx - 2.0f*wx*wy*uxy + wx*wx*uyy;

    float s = (uww>0.0f) ? 1.0f : ((uww<0.0f) ? -1.0f : 0.0f);

    // Upwind |grad| (Osher-Sethian) for the shock term u_t = -s|grad| collapses
    // the ramp ACROSS the edge; the coherence*uvv term diffuses ALONG the edge,
    // straightening the voxel-jittered boundary.
    float g = 0.0f;
    if (s != 0.0f) {
        float Dmx=u-uL, Dpx=uR-u, Dmy=u-uD, Dpy=uU-u;
        if (s>0.0f) {  // erosion
            float a=fmax(Dmx,0.0f), b=fmin(Dpx,0.0f);
            float e=fmax(Dmy,0.0f), f=fmin(Dpy,0.0f);
            g=sqrt(a*a+b*b+e*e+f*f);
        } else {       // dilation
            float a=fmin(Dmx,0.0f), b=fmax(Dpx,0.0f);
            float e=fmin(Dmy,0.0f), f=fmax(Dpy,0.0f);
            g=sqrt(a*a+b*b+e*e+f*f);
        }
    }
    // Gate only the (terracing) shock term; the along-edge diffusion is left
    // ungated — it is ~0 on smooth/planar areas and only straightens edges.
    float upd = u - dt*s*g*w_edge + coherence*uvv;

    // Monotonicity (anti-ringing) constraint.  A shock filter is morphological
    // (dilation/erosion) and must not create a new extremum; clamp the update
    // to the local 3x3 [min,max] so it cannot overshoot into a ringing halo
    // around the edge.  This never limits a legitimate shock (a cell dilating
    // toward the roof value is still allowed to reach it) — it only removes the
    // Gibbs-like overshoot from the discrete + diffusion terms.
    float lo = fmin(fmin(fmin(u, uL), fmin(uR, uD)),
                    fmin(fmin(uU, uLD), fmin(uRD, fmin(uLU, uRU))));
    float hi = fmax(fmax(fmax(u, uL), fmax(uR, uD)),
                    fmax(fmax(uU, uLD), fmax(uRD, fmax(uLU, uRU))));
    u_out[c] = clamp(upd, lo, hi);
}

// ----------------------------------------------------------------
// Gated 3x3 median despeckle of an RGB ortho.  Each pixel is replaced by the
// per-channel median of its valid 3x3 neighbours ONLY when it differs from that
// median by more than `threshold` (max per-channel, 0-255) — so isolated
// speckle is removed while real texture/edges are preserved (an un-gated median
// would soften everything).  No-data cells (valid==0) are passed through and
// never enter a neighbour's median.  in/out: RGB interleaved, 3*ncells.
// 2D dispatch: id(0)=x, id(1)=y.
// ----------------------------------------------------------------
__kernel void ortho_gated_median(
    __global const float* in,
    __global float*       out,
    __global const unsigned char* valid,
    const int   width,
    const int   height,
    const float threshold)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int c = y * width + x;

    float cr = in[3*c+0], cg = in[3*c+1], cb = in[3*c+2];
    if (!valid[c]) { out[3*c+0]=cr; out[3*c+1]=cg; out[3*c+2]=cb; return; }

    float rr[9], gg[9], bb[9];
    int n = 0;
    for (int dy=-1; dy<=1; dy++) {
        int yy=y+dy; if (yy<0 || yy>=height) continue;
        for (int dx=-1; dx<=1; dx++) {
            int xx=x+dx; if (xx<0 || xx>=width) continue;
            int p=yy*width+xx; if (!valid[p]) continue;
            rr[n]=in[3*p+0]; gg[n]=in[3*p+1]; bb[n]=in[3*p+2]; n++;
        }
    }
    // Insertion-sort each channel and take the middle element.
    for (int i=1;i<n;i++){ float v=rr[i]; int j=i; while(j>0&&rr[j-1]>v){rr[j]=rr[j-1];j--;} rr[j]=v; }
    for (int i=1;i<n;i++){ float v=gg[i]; int j=i; while(j>0&&gg[j-1]>v){gg[j]=gg[j-1];j--;} gg[j]=v; }
    for (int i=1;i<n;i++){ float v=bb[i]; int j=i; while(j>0&&bb[j-1]>v){bb[j]=bb[j-1];j--;} bb[j]=v; }
    float mr=rr[n/2], mg=gg[n/2], mb=bb[n/2];

    float d = fmax(fabs(cr-mr), fmax(fabs(cg-mg), fabs(cb-mb)));
    if (d > threshold) { out[3*c+0]=mr; out[3*c+1]=mg; out[3*c+2]=mb; }
    else               { out[3*c+0]=cr; out[3*c+1]=cg; out[3*c+2]=cb; }
})CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
