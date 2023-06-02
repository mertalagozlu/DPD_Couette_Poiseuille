# DPD_Simulation Code - README

This code implements a DPD (Dissipative Particle Dynamics) Simulation in Python using classes to represent particles and their bonds. The simulation can be adjusted to handle different scenarios and various particle types ('F', 'W', 'A', 'B') with different behaviors.

## Code Structure

The main classes of this simulation are:

1. `Particle` - Represents a particle in the simulation, with properties such as position, velocity, force, particle type, and list of bonds. It also contains a method to record the position history of the particle.

2. `Bond` - Represents a bond between two particles, with a method to compute the force exerted by the bond.

3. `DPD_Simulation` - Represents the whole DPD simulation with methods to initialize the particles, compute forces, integrate the equations of motion, create video output, and run the simulation.

## Usage

The main simulation parameters are set in the "main" section at the end of the code:

```python
if __name__ == "__main__":
    L = 15
    rho = 4
    dt = 0.01
    T = 1.0
    gamma = 4.5
    sigma = 1.0
    rc = 1.0
    F_body = np.array([0, 0.3])
    a_C = np.array([[50, 25, 200], [25, 25, 200], [200, 200, 0]])
    steps = 5000
    dpd = DPD_Simulation(L, rho, dt, T, gamma, sigma, rc, a_C, wall_vel=0)
    # dpd.run(steps, F_body)
    # dpd.make_video('part_c', 'part_c/Poiseuille.avi', 22)
```

In this block, you first define the parameters of the simulation and then create an instance of `DPD_Simulation`. Next, you run the simulation with `dpd.run(steps, F_body)`. After that, you can make a video with `dpd.make_video()`.

For example, if you wanted to run the simulation with a different number of steps, you could simply modify the `steps` variable. 

## Required Libraries

This code requires the following libraries:
- os
- cv2
- numpy
- matplotlib

These can be installed with pip:

```bash
pip install opencv-python numpy matplotlib
```

## Notes

The `DPD_Simulation` class includes several simulation settings, such as bond constants, equilibrium bond lengths, and other properties. These settings can be adjusted to suit the specific needs of the simulation. 

The `make_video()` method allows you to create a video representation of your simulation using the output images generated at each step.

Please note that this code includes specific settings and modifications for certain "parts" (A, B, C), and these can be adjusted or removed based on the specific requirements of your simulation.
