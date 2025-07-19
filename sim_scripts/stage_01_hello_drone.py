import asyncio
import math
from omni.isaac.core import SimulationApp, World
from pxr import Gf
import carb

CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)
world = World(stage_units_in_meters=1.0)

async def main():
    await world.reset_async()
    carb.log_info("Simulation initialized")
    
    # Load ground plane
    world.scene.add_default_ground_plane()
    
    # Load drone
    drone_path = "/Isaac/Assets/Parrot_Drone"
    try:
        world.scene.add(drone_path, prim_path="/World/Drone")
        carb.log_info("Drone loaded successfully")
    except Exception as e:
        carb.log_error(f"Failed to load drone: {e}")
        simulation_app.close()
        return
    
    # Simulation loop
    radius, height, speed = 2.0, 1.5, 0.1
    time = 0.0
    while simulation_app.is_running():
        position = Gf.Vec3d(radius * math.cos(time * speed), radius * math.sin(time * speed), height)
        drone = world.scene.get_object("/World/Drone")
        if drone:
            drone.set_world_pose(position=position, orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            carb.log_info(f"Drone position: {position}")
        await world.step_async(render=True)
        time += world.get_physics_dt()
    
    simulation_app.close()

if __name__ == "__main__":
    asyncio.run(main())