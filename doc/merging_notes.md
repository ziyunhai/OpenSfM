# Notes on Multiple Reconstructions Alignment

## Merging

Given a set of reconstructions, let

$$H_a = \begin{pmatrix} s_a R_a & t_a \\ 0 & 1 \end{pmatrix}$$

be the similarity transform mapping points in the global merged reference frame to the local reference frame of reconstruction $a$.

Let $P_{ai} = (R_{ai}\ t_{ai})$ be the projection matrix of camera $i$ in reconstruction $a$, and $P_i = (R_i\ t_i)$ be the projection matrix of camera $i$ in the global reference frame.

The relation between local and global camera positions is:

$$P_i \propto P_{ai} H_a$$

so:

$$s_a (R_i\ t_i) = (R_{ai}\ t_{ai}) \begin{pmatrix} s_a R_a & t_a \\ 0 & 1 \end{pmatrix}$$

Solving for $R_{ai}$ and $t_{ai}$:

$$R_{ai} = R_i R_a^T$$
$$t_{ai} = s_a t_i - R_i R_a^T t_a$$

To find the best absolute projections given the relative ones, minimize:

$$\left\| \log(R_{ai} R_a R_i^T) \right\|^2_{\Sigma_{R_{ai}}} + \left\| t_{ai} - s_a t_i + R_i R_a^T t_a \right\|^2_{\Sigma_{t_{ai}}}$$

with respect to $\{(R_a,\ t_a)\}$.

Alternatively, optimize rotation and translation jointly:

$$\left\| \left(\log(R_{ai} R_a R_i^T),\ t_{ai} - s_a t_i + R_i R_a^T t_a \right) \right\|^2_{\Sigma_{Rt_{ai}}}$$

### Aligning Camera Centers Instead of Translations

Aligning translation vectors directly is problematic: when rotations are misaligned, cameras may end up in different positions even with identical translation vectors. A more robust approach minimizes the distance between optical centers.

Let the optical center of camera $i$ be $o_i = -R_i^T t_i$, and in reconstruction $a$: $o_{ai} = -R_{ai}^T t_{ai}$.

After applying $H_a$, the centers should align:

$$s_a R_a o_i + t_a = o_{ai}$$

Minimize:

$$\left\| s_a R_a o_i + t_a - o_{ai} \right\|^2_{\Sigma_{o_{ai}}}$$

which in terms of $R$s and $t$s is:

$$\left\| R_{ai}^T t_{ai} - s_a R_a R_i^T t_i + t_a \right\|^2_{\Sigma_{o_{ai}}}$$

### Camera Position Prior

The camera center of camera $i$ is $-R_i^T t_i$. To keep it close to GPS position $g_i$, minimize:

$$\left\| g_i + R_i^T t_i \right\|^2_{\Sigma_{g_i}}$$

If camera $i$ is not part of the optimization parameters, add the constraint in terms of $H_a$:

$$\left\| g_i + (R_{ai} R_a)^T (R_{ai} t_a + t_{ai}) / s_a \right\|^2_{\Sigma_{g_i}}$$

### Common Point Constraint

When a point is present in more than one reconstruction, the multiple reconstructions of that point should align. For every pair of reconstructions, let $p_{ai}$ and $p_{bi}$ be point $i$ in reconstructions $a$ and $b$ respectively.

In the global reference frame they should agree:

$$s_a^{-1} R_a^T (p_{ai} - t_a) = s_b^{-1} R_b^T (p_{bi} - t_b)$$

Minimize the difference:

$$\left\| s_a^{-1} R_a^T (p_{ai} - t_a) - s_b^{-1} R_b^T (p_{bi} - t_b) \right\|_{\Sigma_p}$$
