# -*- coding: utf-8 -*-

"""
项目名称: 无人机感知与决策系统 - MVP v1.0
文件名: stage_01_hello_drone.py
版本: 2.1 (持续运行版，用于可视化验证)

功能描述:
    此版本用于可视化验证。它将持续运行仿真，
    直到用户手动关闭Isaac Sim窗口。
"""

import asyncio
import math
import carb

from isaacsim import SimulationApp
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from pxr import Gf

CONFIG = {"headless": False}

async def run_simulation():
    simulation_app = SimulationApp(CONFIG)
    carb.log_info("Isaac Sim App initialized.")

    world = simulation_app.world
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder.")
        simulation_app.close()
        return

    world.scene.add_default_ground_plane()
    drone_asset_path = assets_root_path + "/Isaac/Robots/Parrot/parrot_quadcopter.usd"
    drone_prim_path = "/World/Parrot"
    add_reference_to_stage(usd_path=drone_asset_path, prim_path=drone_prim_path)
    
    await world.reset_async()
    
    drone_prim = XFormPrim(prim_path=drone_prim_path)
    carb.log_info("Drone loaded. Starting flight simulation loop...")
    
    step = 0
    # 核心修正：改回while循环，让仿真持续进行
    while simulation_app.is_running():
        simulation_app.update()

        radius = 2.0
        speed = 1.0
        height = 1.5
        angle = step * speed / 60.0

        position = Gf.Vec3d(radius * math.cos(angle), radius * math.sin(angle), height)
        orientation = Gf.Quatd(1, 0, 0, 0)

        drone_prim.set_world_pose(position, orientation)
        step += 1
    
    # 当用户关闭窗口后，这部分代码会被执行
    carb.log_info("Isaac Sim window was closed.")
    simulation_app.close()
    carb.log_info("Simulation app closed gracefully.")


if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except Exception as e:
        carb.log_error(f"An error occurred: {e}")