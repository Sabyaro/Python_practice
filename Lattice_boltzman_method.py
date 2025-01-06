import jax
import jax.numpy as jnp 
import matplotlib.pyplot as plt 
import cmasher as cmr
from tqdm import tqdm 

N_iterations = 15_000
Re = 80 

N_x = 300
N_y = 50

cylinder_center_x = N_x//5
cylinder_center_y = N_y//2
cylinder_radius_indices = N_y//9
max_horizontal_inflow_velocity = 0.04

Visualize = True
plot_every_n_steps = 100
skip_first_n_iterations = 5000

r"""
LBM Grid: D209
6  2  5
\  |  /
3 -0- 1
/  |  \
7  4   8

"""

N_discrete_velocities = 9

lattice_discrete_velocities = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
])
lattice_indices = jnp.array (
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
)
opposit_lattice_indices = jnp.array (
    [0, 3, 4, 1, 2, 7, 8, 5, 6]
)
lattice_weights = jnp.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])
right_velocities = jnp.array ([1, 5, 8])
left_velocities = jnp.array ([3, 6, 7])
upper_velocities = jnp.array ([2, 5, 6])
bottom_velocities = jnp.array ([4, 7, 8])
vertical_velocities = jnp.array ([0, 2, 4])
horizontal_velocities = jnp.array ([0, 1, 3])

def density(discrete_velocity):
    density_value = jnp.sum(discrete_velocity, axis = -1)

    return density_value

def macroscopic_velocity(discrete_velocity, density_value):
    macroscopic_velocity_value = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete_velocity, 
        lattice_discrete_velocities,)/density_value[..., jnp.newaxis]

    return macroscopic_velocity_value

def equilibrium(macroscopic_velocity_value, density_value):
    projected_discrete_velocity = jnp.einsum(
        "dQ,NMd->NMQ",
        lattice_discrete_velocities,
        macroscopic_velocity_value,
    )

    macroscopic_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocity_value,
        axis=-1,
        ord=2,
    )

    equilibrium_velocity_value =(
        density_value[..., jnp.newaxis] 
        * 
        lattice_weights[jnp.newaxis, jnp.newaxis, :] 
        *
        (
            1 
            + 
            3 * projected_discrete_velocity 
            + 
            9/2 * projected_discrete_velocity**2 
            -
            3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2
        )
    )
    
    return equilibrium_velocity_value

def main(): 
    jax.config.update("jax_enable_x64", True)

    kinematic_viscosity = (
        (
            max_horizontal_inflow_velocity 
            * 
            cylinder_radius_indices
        )/(
            Re
        )
    )
    relaxation_omega = (
        (
        1.0
    )/(
        3.0
        *
        kinematic_viscosity
        +
        0.5
        )
    )
    x = jnp.arange(N_x)
    y = jnp.arange(N_y)
    X, Y = jnp.meshgrid(x,y, indexing = "ij")
    obstacle_mesh = (
        jnp.sqrt(
            (
                X
                -
                cylinder_center_x
            )**2 
            + 
            (
                Y
                -
                cylinder_center_y
            )**2
        )
        <
        cylinder_radius_indices
        )
    velocity_profile = jnp.zeros((N_x, N_y, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(max_horizontal_inflow_velocity)

    @jax.jit

    def update(discrete_velocity_prev):
        discrete_velocity_prev = discrete_velocity_prev.at[-1, : , left_velocities].set(
            discrete_velocity_prev[-2, : , left_velocities]
            )
        density_prev = density(discrete_velocity_prev)
        macroscopic_velocity_prev = macroscopic_velocity(
            discrete_velocity_prev, 
            density_prev,
            )
        
        macroscopic_velocity_prev =\
            macroscopic_velocity_prev.at[0, 1:-1, :].set(
                velocity_profile[0, 1:-1, :]
                )
        density_prev = density_prev.at[0, :].set(
            (
            density(discrete_velocity_prev[0, :, vertical_velocities].T)
            + 
            2*density(discrete_velocity_prev[0, :, left_velocities].T)
            )/ (
                1-macroscopic_velocity_prev[0, : , 0]
                )
            )
        equilibrium_discrete_velocity = equilibrium(
            macroscopic_velocity_prev, 
            density_prev,
            )
        discrete_velocity_prev =\
             discrete_velocity_prev.at[0, :, right_velocities].set(
                 equilibrium_discrete_velocity[0, :, right_velocities]
                 )

        discrete_velocity_post_collision = (
            discrete_velocity_prev 
            - 
            relaxation_omega 
            *
            (discrete_velocity_prev 
             - 
             equilibrium_discrete_velocity)
        )

        for i in range (N_discrete_velocities):
            discrete_velocity_post_collision = \
                discrete_velocity_post_collision.at[obstacle_mesh, lattice_indices[i]].set(
                discrete_velocity_prev[obstacle_mesh, opposit_lattice_indices[i]]
                )
            
        discrete_velocity_stream = discrete_velocity_post_collision
        
        for i in range (N_discrete_velocities):
            discrete_velocity_stream = discrete_velocity_stream.at[:, :, i].set(
            jnp.roll(
                jnp.roll(
                    discrete_velocity_post_collision[:, :, i], 
                    lattice_discrete_velocities[0, i], 
                    axis = 0,
            ),
            lattice_discrete_velocities[1, i], 
            axis=1,
            )
        )
            
        return discrete_velocity_stream

    discrete_velocity_prev = equilibrium(velocity_profile, 
                                         jnp.ones((N_x, N_y)),
                                         )

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6), dpi=100)

        
    for iteration_index in tqdm (range (N_iterations)):
        discrete_velocity_next = update(discrete_velocity_prev)
        discrete_velocity_prev = discrete_velocity_next

        if iteration_index % plot_every_n_steps == 0 and Visualize and iteration_index > skip_first_n_iterations:
                density_value = density(discrete_velocity_next)
                macroscopic_velocity_value = macroscopic_velocity(
                    discrete_velocity_next, 
                    density_value,
                    )
                velocity_magnitude = jnp.linalg.norm(
                    macroscopic_velocity_value, 
                    axis =-1,
                    ord = 2,
                    )
                du_dx, du_dy = jnp.gradient(macroscopic_velocity_value[..., 0])
                dv_dx, dv_dy = jnp.gradient(macroscopic_velocity_value[..., 1])
                curl = (du_dy - dv_dx)

                plt.subplot(211)

                plt.contourf(
                    X, 
                    Y,
                    velocity_magnitude,
                    levels = 50,
                    cmap = cmr.amber,
                )
                plt.colorbar().set_label("Velocity Magnitude")
                plt.gca().add_patch(plt.Circle(
                    (cylinder_center_x,cylinder_center_y), 
                    cylinder_radius_indices, 
                    color = "darkgreen",
                    ))
                
                plt.subplot(212)

                plt.contourf(
                    X, 
                    Y,
                    curl,
                    levels = 50,
                    cmap = cmr.redshift,
                    vmin = 0.02,
                    vmax = 0.02,
                )

                plt.colorbar().set_label("Vorticity Magnitude")
                plt.gca().add_patch(plt.Circle(
                    (cylinder_center_x, cylinder_center_y), 
                    cylinder_radius_indices, 
                    color = "darkgreen",
                    ))
                

                plt.draw()
                plt.pause(0.0001)
                plt.clf()
    if Visualize:
        plt.show()



        


if __name__ == "__main__":
    main()

    


