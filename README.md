# Numerical Simulation of a 2D Electromagnetic Cloak

ME2 Computing Coursework

Notebook: `ME2_CW_Code_Final.ipynb`

## Overview

This project uses the explicit finite difference method to solve the 2D wave equation for an electric field travelling past a cylindrical "invisibility cloak". The goal is to show that a shell with a graded wave speed can guide wavefronts around a reflective object so that they join up again behind it. An add-on Fourier analysis then checks that the wave measured behind the cloak has the same frequency as the source.

## The Physics

### The wave equation

The model uses a single electric field component $E(x, y, t)$, polarised along $z$ and uniform in that direction. The medium has no sources or losses, and only the permittivity varies across the plane. Under these conditions Maxwell's equations reduce exactly to the scalar wave equation:

$$
\frac{\partial^2 E}{\partial t^2} = c^2(x, y) \left( \frac{\partial^2 E}{\partial x^2} + \frac{\partial^2 E}{\partial y^2} \right)
$$

Here $c(x, y) = 1/\sqrt{\mu \varepsilon}$ is the local wave speed. Changing the material changes $c$, and a wavefront bends wherever $c$ changes.

The simulation is non-dimensional, with a background speed of $c_0 = 1$. Axes are labelled in metres and seconds for readability.

### The source

The left edge of the domain is driven with a sinusoid:

$$
E(0, y, t) = \sin(2 \pi f t), \qquad f = 4 \text{ Hz}
$$

This sends an approximately planar wave to the right with wavelength $\lambda = c_0 / f = 0.25$ m. The high frequency packs many wavefronts into the domain, so it is easy to see them bend.

### The object being hidden

A disc of radius $R_1 = 0.3$ m centred at $(1.5, 0)$ has $E = 0$ imposed inside it. This acts like a perfect electric conductor. Without a cloak it would reflect the incoming wave and leave a shadow behind it.

### The cloak

Transformation optics (Pendry, Schurig and Smith, 2006) says that a disc of radius $R_2$ can be squeezed, as a coordinate transformation, into an annulus $R_1 \le r \le R_2$. A material whose properties mimic this transformation makes waves follow the squeezed coordinates, so they pass around the inner region as if it were not there.

A perfect cloak needs anisotropic material properties, which a single scalar wave speed cannot represent. This notebook uses a simplified isotropic graded profile based on the reduced parameter set of Schurig et al. (2006):

$$
c(r) = c_0 \, \frac{R_2}{R_2 - R_1} \left( \frac{r - R_1}{r} \right)^2, \qquad R_1 \le r \le R_2
$$

with $r = \sqrt{(x - 1.5)^2 + y^2}$, $R_1 = 0.3$ m and $R_2 = 0.6$ m.

- At the inner radius, $c \to 0$. Numerically it is clipped to a minimum of $0.01$ so the wave never stops completely.
- At the outer radius, $c = c_0 \cdot 2 \cdot (0.5)^2 = 0.5 \, c_0$.
- Because the speed changes smoothly with $r$, wavefronts in the shell are refracted gradually rather than at one sharp interface.

### Symmetry

The cloak sits on the line $y = 0$, and the source is uniform in $y$, so the solution is symmetric: $E(x, -y, t) = E(x, y, t)$. This means

$$
\left. \frac{\partial E}{\partial y} \right|_{y = 0} = 0
$$

so only the upper half $0 \le y \le 1.5$ is solved, which halves the computation. The result is mirrored for plotting.

## The Maths and Numerical Method

### Discretisation

| Quantity | Value |
|---|---|
| Domain | $0 \le x \le 4$ m, $0 \le y \le 1.5$ m |
| Grid points | $N_x = 400$, $N_y = 150$ |
| Grid spacing | $\Delta x = 4/399 \approx 0.0100$ m, $\Delta y = 1.5/149 \approx 0.0101$ m |
| Points per wavelength (free space) | $\lambda / \Delta x \approx 25$ |
| Time step | $\Delta t \approx 0.00568$ s |
| Number of time steps | $N_t = 1500$ (total time $\approx 8.52$ s) |

Write $E^n_{i,j} \approx E(x_i, y_j, t_n)$. Both the second time derivative and the second space derivatives are replaced by second-order central differences:

$$
\frac{\partial^2 E}{\partial t^2} \approx \frac{E^{n+1}_{i,j} - 2E^n_{i,j} + E^{n-1}_{i,j}}{\Delta t^2},
\qquad
\frac{\partial^2 E}{\partial x^2} \approx \frac{E^n_{i+1,j} - 2E^n_{i,j} + E^n_{i-1,j}}{\Delta x^2}
$$

The $y$ derivative is treated the same way. Substituting into the wave equation and rearranging for the next time level gives the explicit update:

$$
E^{n+1}_{i,j} = 2E^n_{i,j} - E^{n-1}_{i,j}
+ C_x^2 \left( E^n_{i+1,j} - 2E^n_{i,j} + E^n_{i-1,j} \right)
+ C_y^2 \left( E^n_{i,j+1} - 2E^n_{i,j} + E^n_{i,j-1} \right)
$$

with Courant numbers

$$
C_x = \frac{c_{i,j} \, \Delta t}{\Delta x}, \qquad C_y = \frac{c_{i,j} \, \Delta t}{\Delta y}
$$

The error is $O(\Delta t^2, \Delta x^2, \Delta y^2)$. Because $c$ varies in space, $C_x^2$ and $C_y^2$ are calculated once as 2D arrays before the loop. The update uses NumPy array slicing over all interior points, which is much faster than nested loops.

### The first time step

The update needs two previous time levels, but at $t = 0$ only $E^0$ is known. The initial condition $\partial E / \partial t = 0$ is written as a central difference:

$$
\frac{E^1 - E^{-1}}{2 \Delta t} = 0 \quad \Rightarrow \quad E^{-1} = E^1
$$

Substituting this "ghost" value into the update at $n = 0$ gives

$$
E^1_{i,j} = E^0_{i,j}
+ \tfrac{1}{2} C_x^2 \left( E^0_{i+1,j} - 2E^0_{i,j} + E^0_{i-1,j} \right)
+ \tfrac{1}{2} C_y^2 \left( E^0_{i,j+1} - 2E^0_{i,j} + E^0_{i,j-1} \right)
$$

The field starts at zero everywhere, so all of the energy enters through the driven left boundary.

### Stability (CFL condition)

Explicit schemes are only conditionally stable. For the 2D wave equation the condition is

$$
c_{\max} \, \Delta t \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}} \le 1
$$

The code finds $\Delta t_{\max}$ from this condition and uses $\Delta t = 0.8 \, \Delta t_{\max}$ as a safety factor. The cloak speed is also capped at $3 c_0$, so a very fast region can never force $\Delta t$ to become impractically small.

### Boundary conditions

Boundary conditions are reapplied after every update:

| Boundary | Type | Implementation |
|---|---|---|
| Left, $x = 0$ | Driven Dirichlet | $E^{n+1}_{0,j} = \sin(2 \pi f t_{n+1})$ |
| Right, $x = 4$ | Dirichlet | $E^{n+1}_{N_x-1,j} = 0$ |
| Top, $y = 1.5$ | Dirichlet | $E^{n+1}_{i,N_y-1} = 0$ |
| Bottom, $y = 0$ | Neumann (symmetry) | $E^{n+1}_{i,0} = E^{n+1}_{i,1}$, a first-order approximation of $\partial E / \partial y = 0$ |
| Inner object, $r < R_1$ | Dirichlet (conductor) | $E^{n+1} = 0$ |

## Analysis

### Sensor node

A sensor sits directly behind the cloak on the symmetry line, at grid index $i = \lfloor 2.5 / \Delta x \rfloor = 249$, which is $x \approx 2.5$ m, $y = 0$. This is 0.4 m behind the outer edge of the cloak, where an uncloaked object would cast its strongest shadow. The full time history $E(t)$ at this point is recorded.

### Add-on topic: Discrete Fourier Transform

To check that the cloak does not change the frequency of the wave, the sensor signal is transformed into the frequency domain. The early part of the signal, while the wave is still arriving, is discarded. Only the second half is kept: $N = 750$ samples, from $t \approx 4.26$ s onward.

The DFT is implemented directly from its definition, as in Exercises 8, using two nested loops:

$$
c_k = \sum_{n=0}^{N-1} y_n \, e^{-2 \pi i k n / N}, \qquad k = 0, 1, \dots, N-1
$$

Each index $k$ corresponds to a physical frequency

$$
f_k = \frac{k}{N \Delta t}
$$

and the plotted spectral amplitude is $|c_k| / N$. A pure sinusoid of amplitude $A$ gives two peaks of height $A/2$, one at $f$ and its mirror image at $f_s - f$. Only frequencies up to 8 Hz are plotted.

Key properties of the transform in this setup:

| Property | Formula | Value |
|---|---|---|
| Sampling frequency | $f_s = 1 / \Delta t$ | $\approx 176$ Hz |
| Nyquist frequency | $f_s / 2$ | $\approx 88$ Hz |
| Frequency resolution | $\Delta f = 1 / (N \Delta t)$ | $\approx 0.235$ Hz |
| Cost | $O(N^2)$ | $\approx 5.6 \times 10^5$ operations |

### Why the frequency should be preserved

The wave equation used here is linear, and the medium is time invariant: $c$ depends on position only, not on time or on $E$. In a linear time-invariant system each frequency travels independently, and no new frequencies can be created. However much the cloak bends and delays the wave, the signal behind it should therefore contain only the 4 Hz driving frequency. The DFT tests this.

## Results

1. **2D contour plot (mirrored to the full domain).** Shows $E$ at the final time step. Wavefronts pass either side of the cloak and continue behind it, with a non-zero field at the sensor location. The cloak shell is shown in yellow and the conducting object as a hatched grey disc.
2. **3D surface plot (upper half).** Shows the same field as a surface to highlight how the wave amplitude varies across the domain.
3. **Sensor time series.** The field at the sensor stays at zero until the wave arrives at about 2.7 s, then oscillates with growing amplitude.
4. **DFT spectrum.** A single dominant peak sits at 4 Hz, matching the driving frequency marked by the red dashed line. This confirms that the wave behind the cloak keeps the source frequency.

## Assumptions and Limitations

- **Boundary reflections.** The right and top edges are fixed at zero, so they reflect waves back into the domain. A wave reflected from $x = 4$ reaches the sensor at about $t = 5.5$ s, which is inside the DFT window. This does not change the frequency, but it does affect the amplitudes in the time series and the 2D field.
- **Approximate cloak.** The isotropic profile only approximates an ideal transformation optics cloak. The wave speed also jumps from $0.5 c_0$ to $c_0$ at the outer radius, which causes some scattering.
- **Resolution inside the shell.** Near $R_1$ the local wavelength $c/f$ becomes smaller than the grid spacing, so the field in that region is under-resolved.
- **Spectral leakage.** 4 Hz lies at $k \approx 17.04$, not exactly on a frequency bin, so the peak spreads slightly into neighbouring bins.
- **Memory.** The full solution array $E[t, x, y]$ has $1500 \times 400 \times 150$ entries, about 720 MB.

## Running the Notebook

Requirements: Python 3, NumPy, Matplotlib and Jupyter.

```bash
pip install numpy matplotlib jupyter
```

```bash
jupyter notebook ME2_CW_Code_Final.ipynb
```

Run all cells in order. The time stepping and the loop-based DFT both take a few seconds. Make sure around 1 GB of RAM is free for the solution array.

## References

1. J. B. Pendry, D. Schurig and D. R. Smith, "Controlling Electromagnetic Fields", *Science* 312, 1780-1782 (2006).
2. D. Schurig et al., "Metamaterial Electromagnetic Cloak at Microwave Frequencies", *Science* 314, 977-980 (2006).
3. ME2 Computing lecture notes and Exercises 8 (Discrete Fourier Transform).
