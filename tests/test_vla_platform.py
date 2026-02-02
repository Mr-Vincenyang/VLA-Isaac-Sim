# Tests for VLA Platform
"""
VLA平台单元测试
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestActionTokenizer:
    """测试动作分词器"""
    
    def test_encode_decode(self):
        """测试编码解码一致性"""
        from vla_platform.models.action_tokenizer import ActionTokenizer
        
        tokenizer = ActionTokenizer()
        
        # 测试数据
        original_action = np.array([0.0, 0.5, -0.5, 0.1, -0.1, 0.0, 0.8])
        
        # 编码
        tokens = tokenizer.encode(original_action)
        
        # 解码
        decoded_action = tokenizer.decode(tokens)
        
        # 验证（允许量化误差）
        assert np.allclose(original_action, decoded_action, atol=0.01)
    
    def test_normalize(self):
        """测试归一化"""
        from vla_platform.models.action_tokenizer import ActionTokenizer
        
        tokenizer = ActionTokenizer()
        
        low = np.array([-0.05] * 7)
        high = np.array([0.05] * 7)
        
        action = np.array([0.0] * 7)  # 中点
        
        normalized = tokenizer.to_normalized(action, low, high)
        
        assert np.allclose(normalized, np.zeros(7), atol=1e-6)
    
    def test_batch_encode(self):
        """测试批量编码"""
        from vla_platform.models.action_tokenizer import ActionTokenizer
        
        tokenizer = ActionTokenizer()
        
        batch = np.random.uniform(-1, 1, size=(10, 7))
        tokens = tokenizer.encode_batch(batch)
        
        assert tokens.shape == (10, 7)


class TestObservationAction:
    """测试观测和动作数据结构"""
    
    def test_observation_creation(self):
        """测试观测创建"""
        from vla_platform.core.base_interfaces import Observation
        
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        
        obs = Observation(
            image=image,
            joint_positions=np.zeros(7),
            gripper_state=0.5
        )
        
        assert obs.image.shape == (224, 224, 3)
        assert len(obs.joint_positions) == 7
        assert obs.gripper_state == 0.5
    
    def test_action_properties(self):
        """测试动作属性"""
        from vla_platform.core.base_interfaces import Action
        
        action = Action(
            values=np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.8]),
            action_type="delta_ee"
        )
        
        assert np.allclose(action.position_delta, [0.1, 0.2, 0.3])
        assert np.allclose(action.rotation_delta, [0.01, 0.02, 0.03])
        assert action.gripper_action == 0.8


class TestMotionController:
    """测试运动控制器"""
    
    def test_pd_controller(self):
        """测试PD控制器"""
        from vla_platform.control.motion_controller import PDController
        
        kp = np.array([100.0] * 7)
        kd = np.array([10.0] * 7)
        
        controller = PDController(kp, kd)
        
        current_pos = np.zeros(7)
        target_pos = np.ones(7) * 0.1
        current_vel = np.zeros(7)
        
        output = controller.compute(current_pos, target_pos, current_vel)
        
        # 应该有正的控制输出（向目标移动）
        assert np.all(output > 0)
    
    def test_velocity_limit(self):
        """测试速度限制"""
        from vla_platform.control.motion_controller import MotionController
        from vla_platform.core.config import ControlConfig
        
        config = ControlConfig(max_velocity=0.5)
        controller = MotionController(config)
        
        current = np.zeros(7)
        target = np.ones(7) * 10  # 很大的目标
        
        limited = controller.limit_velocity(current, target, dt=0.01)
        
        # 变化应该被限制
        delta = np.abs(limited - current)
        assert np.all(delta <= config.max_velocity * 0.01 + 1e-6)


class TestTrajectoryPlanner:
    """测试轨迹规划器"""
    
    def test_linear_interpolation(self):
        """测试线性插值"""
        from vla_platform.control.trajectory_planner import LinearInterpolator
        
        start = np.zeros(7)
        end = np.ones(7)
        
        trajectory = LinearInterpolator.interpolate(start, end, num_points=11)
        
        assert len(trajectory) == 11
        assert np.allclose(trajectory[0], start)
        assert np.allclose(trajectory[-1], end)
    
    def test_min_jerk_trajectory(self):
        """测试最小急动度轨迹"""
        from vla_platform.control.trajectory_planner import MinJerkTrajectory
        
        start = np.zeros(7)
        end = np.ones(7)
        
        trajectory = MinJerkTrajectory.generate(
            start_pos=start,
            end_pos=end,
            duration=1.0,
            dt=0.1
        )
        
        assert len(trajectory) > 0
        assert np.allclose(trajectory[0].position, start)
        assert np.allclose(trajectory[-1].position, end, atol=0.01)


class TestImpedanceController:
    """测试阻抗控制器"""
    
    def test_impedance_compute(self):
        """测试阻抗控制计算"""
        from vla_platform.control.impedance_controller import ImpedanceController
        
        controller = ImpedanceController()
        
        target_pos = np.array([0.5, 0.0, 0.3])
        controller.set_target(target_pos)
        
        current_pos = np.array([0.4, 0.0, 0.3])  # 偏离目标
        current_quat = np.array([1, 0, 0, 0])
        
        wrench = controller.compute(
            current_position=current_pos,
            current_orientation=current_quat,
            current_velocity=np.zeros(3),
            current_angular_velocity=np.zeros(3)
        )
        
        # x方向应该有正的力（因为current_x < target_x）
        assert wrench[0] > 0


class TestConfig:
    """测试配置管理"""
    
    def test_default_config(self):
        """测试默认配置"""
        from vla_platform.core.config import PlatformConfig, DEFAULT_CONFIG
        
        assert DEFAULT_CONFIG is not None
        assert DEFAULT_CONFIG.remote.port == 8000
        assert DEFAULT_CONFIG.model.action_dim == 7
    
    def test_config_from_yaml(self):
        """测试从YAML加载配置"""
        from vla_platform.core.config import PlatformConfig
        import tempfile
        import yaml
        
        config_data = {
            "remote": {"host": "test-server", "port": 9000},
            "model": {"model_name": "test-model"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = PlatformConfig.from_yaml(temp_path)
            assert config.remote.host == "test-server"
            assert config.remote.port == 9000
            assert config.model.model_name == "test-model"
        finally:
            import os
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
