import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Create a numpy array for a_ij values


class Particle:
    def __init__(self, pos, vel, ptype):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.force = np.zeros(2)
        self.ptype = ptype  # Particle type: 'F', 'W', 'A', 'B'
        self.bonds = []  # List to store bonds
        self.static = False
        self.history = []  # List to store position at each step

    def record_position(self):
        self.history.append(self.pos.copy())  # Note: We need to use copy() because self.pos will be updated

        # Add a property to check if the particle is wall type

    @property
    def is_wall(self):
        return self.ptype == 'W'


class Bond:
    def __init__(self, p1, p2, ks, rs):
        self.p1 = p1
        self.p2 = p2
        self.ks = ks  # Bond constant
        self.rs = rs  # Equilibrium bond length

    def compute_force(self):
        rij = self.p1.pos - self.p2.pos
        r = np.linalg.norm(rij)
        rij_hat = rij / r if r != 0 else 0
        Fs = self.ks * (1 - r / self.rs) * rij_hat
        return Fs


class DPD_Simulation:
    # Remaining part of the class code
    def __init__(self, L, rho, dt, T, gamma, sigma, rc, a, wall_vel=5, num_chains=42, num_rings=10):
        self.num_ring = num_rings
        self.num_chains = num_chains
        self.L = L
        self.rho = rho
        self.dt = dt
        self.T = T
        self.gamma = gamma
        self.sigma = sigma
        self.rc = rc

        self.wall_vel = wall_vel
        self.a = a
        self.particles = self.initialize_particles()

    def make_video(self, image_folder, video_name, fps):

        # get the images
        images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith("step_")]

        # sort the images by the number following "step_"
        images.sort(key=lambda x: int(x[5:-4]))

        # read the first image to get the shape
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'FMP4'
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # write each frame to the video file
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # close the video file
        video.release()

    def apply_body_force(self, F_body):  # FOR PART C
        for p in self.particles:
            if not p.is_wall:  # If the particle is not a wall particle
                p.force += F_body

    def calculate_temperature(self):
        kinetic_energy = sum(0.5 * np.linalg.norm(p.vel) ** 2 for p in self.particles)
        return kinetic_energy / (len(self.particles))

    def calculate_total_momentum(self):
        return sum(p.vel for p in self.particles)

    def initialize_particles(self):
        # Initialization code
        N = int(self.rho * self.L ** 2)  # Number of particles
        positions = np.random.rand(N, 2) * self.L  # Random distribution

        # Check for overlapping particles and regenerate their positions if overlap occurs
        for i in range(N):
            for j in range(i + 1, N):
                if np.array_equal(positions[i], positions[j]):
                    positions[j] = np.random.rand(2) * self.L  # Regenerate position for particle j

        # Initialize velocities to zero
        velocities = np.zeros((N, 2))
        particles = [Particle(pos, vel, 'F') for pos, vel in zip(positions, velocities)]

        # Convert fluid particles within wall region to wall particles for PART B and C
        for p in particles:
            if p.pos[0] < self.rc:
                p.ptype = 'W'
                p.vel = np.array(
                    [0, self.wall_vel])  # Set velocity for left wall particles to move in positive x-direction
                p.static = True  # Set the static property to True for wall particles
            elif p.pos[0] > self.L - self.rc:
                p.ptype = 'W'
                p.vel = np.array(
                    [0, -self.wall_vel])  # Set velocity for right wall particles to move in negative x-direction
                p.static = True  # Set the static property to True for wall particles

        # Select some fluid particles and form chain and ring molecules
        fluid_particles = [p for p in particles if p.ptype == 'F']

        # Ensure there are enough fluid particles to form molecules
        assert len(fluid_particles) >= 16, "Not enough fluid particles to form molecules."

        # # Create a chain molecules for PART B
        # for _ in range(self.num_chains):
        #     if len(fluid_particles) < 7:
        #         break
        #     chain_molecule = fluid_particles[:7]
        #     for i in range(len(chain_molecule)):
        #         chain_molecule[i].ptype = 'A' if i < 2 else 'B'
        #     self.create_molecule(chain_molecule, ks=100, rs=0.1, loop=True)
        #     fluid_particles = fluid_particles[7:]  # Update fluid particles after creating a chain molecule

        # # Create a ring molecule for PART C
        for _ in range(self.num_ring):
            if len(fluid_particles) < 9:
                break
            ring_molecule = fluid_particles[:9]
            for i in range(len(ring_molecule)):
                ring_molecule[i].ptype = 'A'
            self.create_molecule(ring_molecule, ks=100, rs=0.3, ring=True)  # set ring True for part c or set loop True
            fluid_particles = fluid_particles[9:]  # Update fluid particles after creating a chain molecule

        return particles

    def compute_forces(self):
        # PART B
        # ptype_to_index = {'A': 0, 'B': 1, 'F': 2, 'W': 3}
        # PART C
        ptype_to_index = {'A': 0, 'F': 1, 'W': 2}
        # Original forces computation
        for p in self.particles:
            p.force.fill(0.0)

        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                pi, pj = self.particles[i], self.particles[j]

                rij = pi.pos - pj.pos
                rij = rij - self.L * np.round(rij / self.L)  # Minimum image convention for periodic boundary
                r = np.linalg.norm(rij)
                rij_hat = rij / r if r != 0 else np.zeros_like(rij)

                if r < self.rc:
                    # Conservative force
                    # PART A
                    # Fc = self.a * (1 - (r / self.rc)) * rij_hat
                    # PART B or C
                    Fc = self.a[ptype_to_index[pi.ptype]][ptype_to_index[pj.ptype]] * (1 - (r / self.rc)) * rij_hat

                    # Dissipative force
                    vij = pi.vel - pj.vel
                    wD = (1 - (r / self.rc)) ** 2
                    Fd = -self.gamma * wD * r * np.dot(rij_hat, vij) * rij_hat

                    # Random force
                    xi = np.random.randn()
                    wR = 1 - (r / self.rc)
                    Fr = self.sigma * wR * r * xi * rij_hat * np.sqrt(self.dt)

                    F = Fc + Fd + Fr

                    pi.force += F
                    pj.force -= F

        # Additional bond forces
        for p in self.particles:
            for bond in p.bonds:
                Fb = bond.compute_force()
                if bond.p1 == p:
                    p.force += Fb
                else:
                    p.force -= Fb

    def create_molecule(self, molecule_particles, ks, rs, ring=False, loop=False):
        prev_particle = None
        for particle in molecule_particles:
            if prev_particle is not None:
                bond = Bond(prev_particle, particle, ks, rs)
                prev_particle.bonds.append(bond)
                particle.bonds.append(bond)

            prev_particle = particle

            # If the molecule is a ring or a looped chain, create a bond between the last and first particle
            if ring or loop:
                bond = Bond(particle, molecule_particles[0], ks, rs)
                particle.bonds.append(bond)
                molecule_particles[0].bonds.append(bond)

    def integrate(self):
        for p in self.particles:
            p.record_position()
            if p.is_wall:  # If the particle is a wall particle
                # FOR PART A
                # p.pos += self.dt * p.vel + 0.5 * self.dt ** 2 * p.force
                # p.vel += 0.5 * self.dt * p.force

                # FOR PART B: Ignore the x-component of the force and only update y-position
                # p.pos[1] += self.dt * p.vel[1] + 0.5 * self.dt ** 2 * p.force[1]

                # FOR PART C: We dont need iteration on position of wall at all

                p.pos = p.pos % self.L

            else:  # If the particle is not a wall particle
                p.pos += self.dt * p.vel + 0.5 * self.dt ** 2 * p.force
                p.pos = p.pos % self.L  # Apply periodic boundary conditions
                p.vel += 0.5 * self.dt * p.force

        self.compute_forces()

    def update_plot(self, i):
        # plt.clf()  # clear the plot
        if i % 10 == 0:
            for p in self.particles:
                if p.ptype == 'F':
                    plt.scatter(*p.pos, color='blue')
                elif p.ptype == 'A':
                    plt.scatter(*p.pos, color='green')
                elif p.ptype == 'B':
                    plt.scatter(*p.pos, color='red')
                elif p.ptype == 'W':
                    plt.scatter(*p.pos, color='black')

            plt.xlim(0, self.L)
            plt.ylim(0, self.L)
            plt.title(f"Step: {i}")
            plt.savefig(f'part_c/step_{i}.png')  # save plot as an image
            plt.close()  # close the figure

    def run(self, steps, F_body):
        # fig = plt.figure()
        tempeature_list = []
        momentum_list = []
        if not os.path.exists('part_c'):
            os.makedirs('part_c')

        for i in range(steps):
            # PART C apply constant body force at each iteration to each non-wall particle
            self.apply_body_force(F_body)
            self.integrate()
            self.update_plot(i)

            # anim = FuncAnimation(fig, animate, frames=steps, repeat=False)
            # plt.show()

            # Calculate and print temperature and total momentum at each step
            temperature = self.calculate_temperature()
            total_momentum = self.calculate_total_momentum()
            tempeature_list.append(temperature)
            momentum_list.append(total_momentum)
            print(f"Step {i}, Temperature: {temperature}, Total Momentum: {np.linalg.norm(total_momentum)}")

        plt.plot(np.arange(0, steps, 1), tempeature_list)
        plt.title(f"Step vs Temperature")
        plt.savefig(f'part_c/step vs temperature.png')
        plt.close()

        plt.plot(np.arange(0, steps, 1), momentum_list)
        plt.title(f"Step vs Momentum")
        plt.savefig(f'part_c/step vs momentum.png')
        plt.close()


if __name__ == "__main__":
    L = 15
    rho = 4
    dt = 0.01
    T = 1.0
    gamma = 4.5
    sigma = 1.0
    rc = 1.0
    # part A
    # a_A = 25.0
    # part B
    # a_B = np.array([
    #     [50, 25, 25, 200],
    #     [25, 1, 300, 200],
    #     [25, 300, 25, 200],
    #     [200, 200, 200, 0]
    # ])
    # PART C Constant force in y direction
    F_body = np.array([0, 0.3])
    a_C = np.array([[50, 25, 200], [25, 25, 200], [200, 200, 0]])
    steps = 5000  # set 5000 for part c

    dpd = DPD_Simulation(L, rho, dt, T, gamma, sigma, rc, a_C, wall_vel=0)  # set wall velocity 0 for PART C
    # dpd.run(steps, F_body)
    #dpd.make_video('part_b', 'part_b/Couette.avi', 10)  # uncomment if you want a video of the steps
    #dpd.make_video('part_c', 'part_c/Poiseuille.avi', 22) #uncomment if you want a video of the steps
