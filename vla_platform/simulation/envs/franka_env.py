# VLA Platform - Franka Panda Environment
"""
Franka Panda机械臂仿真环境
用于桌面抓取任务
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging

# Isaac Sim imports - 支持新旧两种命名空间
ISAAC_SIM_AVAILABLE = False
Franka = None
RigidPrim = None
XFormPrim = None
DynamicCuboid = None
ArticulationView = None
prim_utils = None
get_assets_root_path = None

# 尝试新的 isaacsim 命名空间 (Isaac Sim 5.x)
try:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim, SingleXFormPrim as XFormPrim
    from isaacsim.core.api.objects import DynamicCuboid
    # Fix: 正确的导入路径是 isaacsim.robot.manipulators.examples.franka
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.core.prims import ArticulationView
    import isaacsim.core.utils.prims as prim_utils
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    pass

# 尝试旧的 omni.isaac 命名空间 (兼容性)
if not ISAAC_SIM_AVAILABLE:
    try:
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.prims import RigidPrim, XFormPrim
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.franka import Franka
        from omni.isaac.core.articulations import ArticulationView
        import omni.isaac.core.utils.prims as prim_utils
        ISAAC_SIM_AVAILABLE = True
    except ImportError:
        pass

from vla_platform.core.base_interfaces import RobotController, Observation, Action
from vla_platform.core.config import SimulationConfig
from vla_platform.simulation.sim_manager import SimulationManager

logger = logging.getLogger(__name__)


@dataclass
class FrankaEnvConfig:
    """Franka环境配置"""
    robot_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    robot_orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    table_position: List[float] = field(default_factory=lambda: [0.4, 0.0, 0.0])
    table_size: List[float] = field(default_factory=lambda: [0.6, 1.0, 0.05])
    
    # 目标物体配置
    num_objects: int = 1
    object_colors: List[List[float]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    object_size_range: Tuple[float, float] = (0.03, 0.06)
    
    # 工作空间边界
    workspace_x: Tuple[float, float] = (0.25, 0.55)
    workspace_y: Tuple[float, float] = (-0.3, 0.3)
    workspace_z: Tuple[float, float] = (0.02, 0.3)


class FrankaGraspEnv(RobotController):
    """
    Franka Panda抓取环境
    
    提供用于VLA模型的桌面抓取仿真环境
    """
    
    # Franka Panda默认关节位置
    DEFAULT_JOINT_POSITIONS = np.array([
        0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,  # 7个臂关节
        0.04, 0.04  # 2个夹爪关节
    ])
    
    # Franka关节限制
    JOINT_LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    def __init__(
        self,
        sim_manager: SimulationManager,
        config: Optional[FrankaEnvConfig] = None,
        camera_manager = None
    ):
        """
        初始化Franka抓取环境
        
        Args:
            sim_manager: 仿真管理器
            config: 环境配置
            camera_manager: 相机管理器（可选）
        """
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError("Isaac Sim not available")
        
        self.sim_manager = sim_manager
        self.config = config or FrankaEnvConfig()
        self.camera_manager = camera_manager
        
        self._robot: Optional[Franka] = None
        self._objects: List[RigidPrim] = []
        self._table: Optional[RigidPrim] = None
        self._is_initialized = False
        
    @property
    def robot(self) -> Optional[Franka]:
        return self._robot
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def setup(self) -> None:
        """设置环境"""
        world = self.sim_manager.world
        if world is None:
            raise RuntimeError("Simulation world not created")
        
        # 添加Franka机械臂
        self._add_robot()
        
        # 添加桌子
        self._add_table()
        
        # 添加目标物体
        self._add_objects()
        
        # 重置世界
        world.reset()
        
        self._is_initialized = True
        logger.info("Franka grasp environment setup complete")
    
    def _add_robot(self) -> None:
        """添加Franka机械臂"""
        assets_root = get_assets_root_path()
        
        self._robot = Franka(
            prim_path="/World/Franka",
            name="franka",
            position=np.array(self.config.robot_position),
            orientation=np.array(self.config.robot_orientation),
        )
        
        self.sim_manager.world.scene.add(self._robot)
        logger.info("Added Franka robot")
    
    def _add_table(self) -> None:
        """添加桌子"""
        table_prim_path = "/World/Table"
        
        # 计算桌子位置 (3D)
        table_pos = [
            self.config.table_position[0],
            self.config.table_position[1],
            self.config.table_size[2] / 2  # z轴位置
        ]
        
        # Isaac Sim 5.x: size 是标量，用 scale 调整不同方向的尺寸
        base_size = 1.0  # 基础边长
        scale = np.array([
            self.config.table_size[0] / base_size,
            self.config.table_size[1] / base_size,
            self.config.table_size[2] / base_size
        ])
        
        self._table = DynamicCuboid(
            prim_path=table_prim_path,
            name="table",
            position=np.array(table_pos),
            size=base_size,  # 标量
            scale=scale,
            color=np.array([0.5, 0.35, 0.2]),  # 木色
            mass=1000.0  # 重量很大使其固定
        )
        
        self.sim_manager.world.scene.add(self._table)
        logger.info("Added table")
    
    def _add_objects(self) -> None:
        """添加抓取目标物体"""
        self._objects.clear()
        
        for i in range(self.config.num_objects):
            # 随机位置
            x = np.random.uniform(*self.config.workspace_x)
            y = np.random.uniform(*self.config.workspace_y)
            z = self.config.table_size[2] + 0.05  # 桌面上方
            
            # 随机大小 (标量)
            size = float(np.random.uniform(*self.config.object_size_range))
            
            # 颜色
            color = self.config.object_colors[i % len(self.config.object_colors)]
            
            obj = DynamicCuboid(
                prim_path=f"/World/Object_{i}",
                name=f"object_{i}",
                position=np.array([x, y, z]),
                size=size,  # 标量浮点数
                color=np.array(color),
                mass=0.1
            )
            
            self.sim_manager.world.scene.add(obj)
            self._objects.append(obj)
        
        logger.info(f"Added {self.config.num_objects} objects")
    
    def get_joint_positions(self) -> np.ndarray:
        """获取关节位置"""
        if self._robot is None:
            return np.zeros(9)
        return self._robot.get_joint_positions()
    
    def get_joint_velocities(self) -> np.ndarray:
        """获取关节速度"""
        if self._robot is None:
            return np.zeros(9)
        return self._robot.get_joint_velocities()
    
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器位姿
        
        Returns:
            (position, quaternion)
        """
        if self._robot is None:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        # 获取末端执行器的世界位姿
        ee_prim = self._robot.gripper
        if ee_prim is not None:
            pos, quat = ee_prim.get_world_pose()
            return pos, quat
        
        return np.zeros(3), np.array([1, 0, 0, 0])
    
    def get_gripper_state(self) -> float:
        """
        获取夹爪状态
        
        Returns:
            0.0 (完全闭合) 到 1.0 (完全张开)
        """
        if self._robot is None:
            return 0.0
        
        joint_positions = self._robot.get_joint_positions()
        # Franka夹爪关节索引是7和8
        gripper_pos = (joint_positions[7] + joint_positions[8]) / 2
        # 归一化到[0, 1]，最大张开是0.04
        return np.clip(gripper_pos / 0.04, 0.0, 1.0)
    
    def get_observation(self) -> Observation:
        """获取当前观测"""
        # 获取相机图像
        if self.camera_manager is not None:
            camera_data = self.camera_manager.capture()
            image = camera_data.get("rgb", np.zeros((224, 224, 3), dtype=np.uint8))
            depth = camera_data.get("depth", None)
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            depth = None
        
        # 获取机器人状态
        joint_positions = self.get_joint_positions()
        joint_velocities = self.get_joint_velocities()
        ee_position, ee_orientation = self.get_ee_pose()
        gripper_state = self.get_gripper_state()
        
        return Observation(
            image=image,
            depth=depth,
            joint_positions=joint_positions[:7],  # 只返回臂关节
            joint_velocities=joint_velocities[:7],
            ee_position=ee_position,
            ee_orientation=ee_orientation,
            gripper_state=gripper_state
        )
    
    def apply_action(self, action: Action) -> bool:
        """
        应用动作到机器人
        
        Args:
            action: VLA模型输出的动作
            
        Returns:
            是否成功应用
        """
        if self._robot is None:
            return False
        
        try:
            if action.action_type == "delta_ee":
                # 末端执行器增量控制
                return self._apply_delta_ee_action(action)
            elif action.action_type == "absolute_ee":
                # 末端执行器绝对位置控制
                return self._apply_absolute_ee_action(action)
            elif action.action_type == "joint":
                # 关节空间控制
                return self._apply_joint_action(action)
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to apply action: {e}")
            return False
    
    def _apply_delta_ee_action(self, action: Action) -> bool:
        """应用末端执行器增量动作"""
        # 获取当前位姿
        current_pos, current_quat = self.get_ee_pose()
        
        # 计算目标位姿
        target_pos = current_pos + action.position_delta
        
        # 简化处理：直接使用IK控制器（如果可用）
        # 这里先使用关节空间近似控制
        # 实际应用中应使用逆运动学
        
        # 处理夹爪动作
        gripper_action = action.gripper_action
        self._set_gripper(gripper_action)
        
        return True
    
    def _apply_absolute_ee_action(self, action: Action) -> bool:
        """应用末端执行器绝对位置动作"""
        target_pos = action.values[:3]
        # 使用逆运动学计算目标关节位置
        # 简化实现
        self._set_gripper(action.gripper_action)
        return True
    
    def _apply_joint_action(self, action: Action) -> bool:
        """应用关节空间动作"""
        if len(action.values) >= 7:
            joint_positions = action.values[:7]
            # 裁剪到关节限制
            joint_positions = np.clip(
                joint_positions,
                self.JOINT_LIMITS_LOW,
                self.JOINT_LIMITS_HIGH
            )
            self._robot.set_joint_positions(joint_positions)
        
        if len(action.values) > 7:
            self._set_gripper(action.values[7] if len(action.values) > 7 else 0.0)
        
        return True
    
    def _set_gripper(self, gripper_value: float) -> None:
        """
        设置夹爪状态
        
        Args:
            gripper_value: 0.0 (闭合) 到 1.0 (张开)
        """
        if self._robot is None:
            return
        
        target_width = np.clip(gripper_value, 0.0, 1.0) * 0.04
        current_positions = self._robot.get_joint_positions()
        current_positions[7] = target_width
        current_positions[8] = target_width
        self._robot.set_joint_positions(current_positions)
    
    def reset(self, initial_config: Optional[Dict[str, Any]] = None) -> Observation:
        """
        重置环境
        
        Args:
            initial_config: 可选的初始配置
            
        Returns:
            初始观测
        """
        # 重置机器人到默认位置
        if self._robot is not None:
            self._robot.set_joint_positions(self.DEFAULT_JOINT_POSITIONS)
        
        # 随机化物体位置
        if initial_config and initial_config.get("randomize_objects", True):
            self._randomize_objects()
        
        # 运行几步让物理稳定
        for _ in range(10):
            self.sim_manager.step(render=False)
        
        return self.get_observation()
    
    def _randomize_objects(self) -> None:
        """随机化物体位置"""
        for obj in self._objects:
            x = np.random.uniform(*self.config.workspace_x)
            y = np.random.uniform(*self.config.workspace_y)
            z = self.config.table_size[2] + 0.05
            obj.set_world_pose(position=np.array([x, y, z]))
    
    def get_object_positions(self) -> List[np.ndarray]:
        """获取所有物体位置"""
        positions = []
        for obj in self._objects:
            pos, _ = obj.get_world_pose()
            positions.append(pos)
        return positions
    
    def check_grasp_success(self, object_index: int = 0) -> bool:
        """
        检查抓取是否成功
        
        简单判断：物体是否被抬起
        """
        if object_index >= len(self._objects):
            return False
        
        obj = self._objects[object_index]
        pos, _ = obj.get_world_pose()
        
        # 如果物体高于一定高度，认为抓取成功
        return pos[2] > self.config.table_size[2] + 0.1
