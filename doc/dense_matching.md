# Dense Matching Notes

Mathematical notes on the dense matching module.

## Backprojection at a Given Depth

The backprojection of a pixel $q = (q_x, q_y, 1)^T$ at depth $d$ in camera coordinates is:

$$X = d K^{-1} q$$

## Backprojection to a Plane

The backprojection of a pixel $q = (q_x, q_y, 1)^T$ onto the plane $\pi = (v^T, 1)$ is:

$$X = \frac{-K^{-1} q}{v^T K^{-1} q}$$

with depth:

$$d = \frac{-1}{v^T K^{-1} q}$$

## Plane Given Point and Normal

The plane

$$\pi = \left( \frac{-n^T}{n^T X},\ 1 \right)$$

contains point $X$ and has normal $n$.

## Plane of Constant Depth

A plane of constant depth $d$ is defined by $z = d$ in camera coordinates:

$$\pi_c = (0, 0, -1/d, 1)$$

## Plane Coordinates Conversion

A plane's coordinates in world and camera frames are related by:

$$\pi_w = \begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix} \pi_c$$

## Plane-Induced Homography

Given a plane in camera coordinates $\pi_c = (v^T, 1)$, the homography from image 1 to image 2 is:

$$H = K_2 \left[R_2 R_1^T + (R_2 R_1^T t_1 - t_2)\, v^T\right] K_1^{-1}$$

Pre-computing:

$$Q_{12} = R_2 R_1^T, \quad a_{12} = R_2 R_1^T t_1 - t_2$$

gives:

$$H = K_2 \left[Q_{12} + a_{12}\, v^T\right] K_1^{-1}$$

## Local Affine Approximation of a Homography

The homography mapping defined by matrix $H$ is:

$$f(x, y) = \begin{pmatrix} u/w \\ v/w \end{pmatrix}$$

where $u = H_1 (x, y, 1)^T$, $v = H_2 (x, y, 1)^T$, $w = H_3 (x, y, 1)^T$.

The differential is:

$$Df(x, y) = \frac{1}{w^2}
\begin{pmatrix}
   H_{11} w - H_{31} u & H_{12} w - H_{32} u \\
   H_{21} w - H_{31} v & H_{22} w - H_{32} v
\end{pmatrix}$$

The linear approximation around $(x_0, y_0)$ is:

$$f(x_0 + dx,\ y_0 + dy) = f(x_0, y_0) + Df(x_0, y_0)(dx, dy)^T$$

## Undistortion

The dense module assumes perspective images with no radial distortion. For perspective images, undistorted versions are generated using estimated distortion parameters $k_1$, $k_2$.

Spherical (360°) images cannot be unwarped into a single perspective view. Multiple perspective views are generated to cover the full field of view.

Undistortion is therefore a process that takes a reconstruction as input and produces a new reconstruction as output. The input may contain radially distorted images and panoramas; the output contains only undistorted perspective images. A new track graph is also generated.
