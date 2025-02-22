import genesis as gs


########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound=(-0.025, -0.025, 0.0),
        upper_bound=(0.025, 0.025, 0.05),
        particle_size=0.01,
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary=True,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

liquid = scene.add_entity(
    # viscous liquid
    # material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Cylinder(
        pos=(0.0, 0.0, 0.025),  # Lowered initial position
        height=0.04,  # Reduced height to fit inside boundary
        radius=0.025,  # Adjusted radius for better containment
    ),
    surface=gs.surfaces.Default(
        color=(0.4, 0.8, 1.0),
        vis_mode="particle",
        # vis_mode="recon",
    ),
)

########################## build ##########################
scene.build()

horizon = 1000
for i in range(horizon):
    scene.step()

# get particle positions
particles = liquid.get_particles()
