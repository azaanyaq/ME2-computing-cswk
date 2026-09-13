# Simulating an Invisibility Cloak for Waves

<img width="1245" height="793" alt="Screenshot 2026-09-13 at 6 09 53 pm" src="https://github.com/user-attachments/assets/a08c11ab-c63c-4cd7-a668-93c3afe583f4" />


*The electric field at the end of the simulation (t = 8.52 s). Red and blue show positive and negative field. The grey hatched disc is the object, the yellow ring is the cloak and the cyan dot is the sensor.*

ME2 Computing Coursework

Notebook: `ME2_CW_Code_Final.ipynb`

## What This Project Shows

A wave hits an object. Normally the object blocks the wave and leaves a "shadow" behind it.

This project surrounds the object with a special ring (a **cloak**) that changes how fast the wave travels. The ring guides the wave around the object so it joins back up on the other side.

The notebook:

1. Simulates the wave moving past the cloaked object.
2. Records the wave at a point directly behind the object.
3. Uses a Fourier transform to check that the wave behind the object has the same frequency as the wave that was sent in.

## The Physics

### How the wave moves

The electric field $E$ follows the **2D wave equation**:

```math
\frac{\partial^2 E}{\partial t^2} = c^2 \left( \frac{\partial^2 E}{\partial x^2} + \frac{\partial^2 E}{\partial y^2} \right)
```

In simple terms, how quickly the field speeds up or slows down at a point depends on how curved the wave is around that point. $c$ is the wave speed. Where $c$ changes, the wave bends.

### The wave source

The left edge of the domain moves up and down like a sine wave:

```math
E = \sin(2 \pi f t), \qquad f = 4 \text{ Hz}
```

This sends waves travelling to the right with a wavelength of $\lambda = c / f = 0.25$ m.

### The object

The object is a circle of radius 0.3 m at $(1.5, 0)$. The field inside it is forced to zero ($E = 0$), so it acts like a metal cylinder that reflects waves.

### The cloak

The cloak is a ring around the object, from $R_1 = 0.3$ m to $R_2 = 0.6$ m. Inside the ring the wave speed changes with distance $r$ from the centre:

```math
c(r) = c_0 \frac{R_2}{R_2 - R_1} \left( \frac{r - R_1}{r} \right)^2
```

- Next to the object ($r = R_1$) the wave is almost stopped.
- At the outside of the ring ($r = R_2$) the wave travels at half its normal speed.
- Because the speed changes smoothly, the wave bends gradually as it passes through the ring.

This idea comes from **transformation optics** (Pendry et al., 2006). A perfect cloak needs more complex materials, so this is a simplified version.

### Using symmetry

The setup looks the same above and below the line $y = 0$. So only the top half is calculated, which halves the work, and the result is mirrored for the plots. On the symmetry line the wave has zero slope:

```math
\frac{\partial E}{\partial y} = 0 \quad \text{at } y = 0
```

## The Numerical Method

### Setting up the grid

The domain (4 m by 1.5 m) is split into a grid of 400 by 150 points, about 1 cm apart. Time moves forward in small steps of about 0.0057 s, for 1500 steps (8.5 s in total).

### Finite differences

A computer can't handle derivatives directly, so each one is replaced by a **central difference**, which uses neighbouring points. For example:

```math
\frac{\partial^2 E}{\partial x^2} \approx \frac{E_{i+1} - 2E_i + E_{i-1}}{\Delta x^2}
```

Doing this for every derivative in the wave equation and rearranging gives a formula for the field at the **next** time step:

```math
E^{n+1}_{i,j} = 2E^n_{i,j} - E^{n-1}_{i,j} + C_x^2 \left( E^n_{i+1,j} - 2E^n_{i,j} + E^n_{i-1,j} \right) + C_y^2 \left( E^n_{i,j+1} - 2E^n_{i,j} + E^n_{i,j-1} \right)
```

where

```math
C_x = \frac{c \Delta t}{\Delta x}, \qquad C_y = \frac{c \Delta t}{\Delta y}
```

In words, the new value at a point depends on its value at the last two time steps and on the values at its four neighbours. The code applies this formula to every point at once using NumPy array slicing, which is much faster than loops.

### The first step

The formula needs **two** earlier time steps, but at the start there is only one. The wave starts at rest ($\partial E / \partial t = 0$), which gives a special first step:

```math
E^1_{i,j} = E^0_{i,j} + \frac{1}{2} C_x^2 \left( E^0_{i+1,j} - 2E^0_{i,j} + E^0_{i-1,j} \right) + \frac{1}{2} C_y^2 \left( E^0_{i,j+1} - 2E^0_{i,j} + E^0_{i,j-1} \right)
```

### Keeping it stable

If the time step is too big, the simulation blows up. It stays stable as long as

```math
c_{\max} \Delta t \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}} \le 1
```

The code works out the largest time step allowed and then uses 80% of it to be safe.

### Boundary conditions

| Edge | What happens |
|---|---|
| Left | Sine wave source |
| Right and top | Field fixed at zero |
| Bottom (symmetry line) | Zero slope, set by copying the row above: $E_{i,0} = E_{i,1}$ |
| Inside the object | Field fixed at zero |

## Analysis

### Measuring the wave behind the cloak

A **sensor** is placed at $x = 2.5$ m, $y = 0$, directly behind the cloak, where the shadow would normally be. The field at this point is recorded at every time step.

### Fourier analysis (add-on topic)

A **Discrete Fourier Transform (DFT)** breaks a signal into the frequencies it contains. It shows whether the wave behind the cloak is still vibrating at 4 Hz.

Only the second half of the recording is used (from about 4.3 s onward), after the wave has had time to arrive and settle.

The DFT is written by hand, as in Exercises 8:

```math
c_k = \sum_{n=0}^{N-1} y_n e^{-2 \pi i k n / N}
```

- $y_n$ are the $N$ recorded values.
- $c_k$ measures how much of frequency number $k$ is in the signal.

Each $k$ is turned into a real frequency in Hz using

```math
f_k = \frac{k}{N \Delta t}
```

and the size of each frequency is plotted as $|c_k| / N$.

With $N = 750$ samples, the spectrum has a spacing of about 0.23 Hz between frequency points.

### Why the frequency should stay the same

The cloak changes the wave's **speed and direction**, but it doesn't change over time and doesn't depend on how big the wave is. A material like that can't create new frequencies. So if the simulation is working, the only frequency behind the cloak should be the 4 Hz that was sent in.

## Results

1. **2D colour plot.** Shows the waves at the end of the simulation. The waves pass around the cloak and carry on behind it.
2. **3D surface plot.** Shows the same wave as a surface, so its height is easier to see.
3. **Sensor signal over time.** The sensor reads zero until the wave arrives at about 2.7 s, then it oscillates.
4. **Frequency spectrum.** There is one clear peak at **4 Hz**, the same as the source. This confirms the cloak does not change the wave's frequency.

## Limitations

- The right and top edges reflect waves back into the domain, and some of these reflections reach the sensor.
- The cloak is a simplified version, so some of the wave still scatters.
- Near the object the wave slows down so much that the grid is too coarse to capture it accurately.
- The full results array is large (about 720 MB of memory).

## How to Run

```bash
pip install numpy matplotlib jupyter
```

```bash
jupyter notebook ME2_CW_Code_Final.ipynb
```

Run all the cells from top to bottom.

## References

1. J. B. Pendry, D. Schurig and D. R. Smith, "Controlling Electromagnetic Fields", *Science* 312, 1780-1782 (2006).
2. ME2 Computing lecture notes and Exercises 8 (Discrete Fourier Transform).
