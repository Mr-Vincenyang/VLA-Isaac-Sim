# VLA Server Deployment Script
"""
远程服务器VLA模型部署脚本

使用方法:
    python server_deploy.py --model openvla/openvla-7b --port 8000
    
依赖:
    pip install -r requirements_server.txt
"""
import argparse
import logging
from pathlib import Path
from typing import Optional
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局模型变量
model = None
processor = None
device = None


def load_model(
    model_name: str = "openvla/openvla-7b",
    quantization: Optional[str] = None,
    device_map: str = "auto"
):
    """
    加载VLA模型
    
    Args:
        model_name: HuggingFace模型名称
        quantization: 量化选项 ("int4", "int8", None)
        device_map: 设备映射策略
    """
    global model, processor, device
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        # 配置量化
        # 注意：不使用bfloat16以避免数据类型不匹配问题
        # 使用float32确保兼容性，或根据量化选项调整
        if quantization == "int8":
            load_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.float32,  # int8量化时使用float32
                "load_in_8bit": True,
            }
        elif quantization == "int4":
            load_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.float32,  # int4量化时使用float32
                "load_in_4bit": True,
            }
        else:
            # 无量化时使用float32以确保兼容性
            # 如果显存充足，可以使用torch.float16或torch.bfloat16
            # 但为避免数据类型错误，默认使用float32
            load_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.float32,
            }
        
        # 加载模型和处理器
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            **load_kwargs
        )
        
        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def decode_image(image_data: str) -> Image.Image:
    """解码Base64图像"""
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def predict_action(
    image: Image.Image,
    instruction: str,
    temperature: float = 0.0,
    **kwargs
) -> np.ndarray:
    """
    预测机器人动作
    
    Args:
        image: 输入图像
        instruction: 语言指令
        temperature: 采样温度
        
    Returns:
        7维动作向量
    """
    global model, processor, device
    
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")
    
    # OpenVLA特定：构建正确的输入格式
    # OpenVLA期望的格式是：<image>instruction
    prompt = f"<image>{instruction}"
    
    # 预处理
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )
    
    # 确保输入数据类型与模型一致（float32）
    # 并移动到正确的设备
    for key in inputs:
        if torch.is_floating_point(inputs[key]):
            inputs[key] = inputs[key].to(device, dtype=torch.float32)
        else:
            inputs[key] = inputs[key].to(device)
    
    # 推理
    with torch.no_grad():
        # OpenVLA使用特定的生成参数
        generate_kwargs = {
            "max_new_tokens": 7,  # OpenVLA输出7个动作token
            "do_sample": temperature > 0,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        
        outputs = model.generate(**inputs, **generate_kwargs)
    
    # 解码动作token
    # OpenVLA输出的是动作token，需要转换为实际的动作值
    action = decode_openvla_action(outputs, processor)
    
    return action


def decode_openvla_action(outputs, processor) -> np.ndarray:
    """
    解码OpenVLA输出的动作
    
    OpenVLA输出7个token，每个token对应一个动作维度
    需要将token ID转换为实际的动作值
    
    Args:
        outputs: 模型生成的输出
        processor: 处理器
        
    Returns:
        7维动作向量
    """
    try:
        # 获取最后7个token（动作token）
        action_tokens = outputs[0, -7:].cpu().numpy()
        
        # OpenVLA使用256个离散化的动作bin
        # 将token ID转换为实际动作值（范围通常是-1到1）
        # 每个维度是独立的，token ID 0-255映射到-1.0到1.0
        action = np.zeros(7)
        for i in range(7):
            # 将token ID映射到动作值
            # token 0-255 -> -1.0到1.0
            token_id = action_tokens[i]
            action[i] = (token_id / 127.5) - 1.0
        
        return action
        
    except Exception as e:
        logger.error(f"Failed to decode action: {e}")
        # 返回零动作作为后备
        return np.zeros(7)


def parse_action_from_text(text: str) -> np.ndarray:
    """
    从生成的文本中解析动作
    
    OpenVLA输出通常是包含动作数组的文本
    尝试提取其中的7个数字作为动作
    
    Args:
        text: 模型生成的文本
        
    Returns:
        7维动作向量
    """
    import re
    
    # 尝试找到方括号中的数组
    bracket_match = re.search(r'\[([\d\s,\.\-]+)\]', text)
    if bracket_match:
        try:
            # 提取方括号内的内容
            array_str = bracket_match.group(1)
            # 分割并转换为浮点数
            numbers = [float(x.strip()) for x in array_str.split(',')]
            if len(numbers) >= 7:
                return np.array(numbers[:7])
        except:
            pass
    
    # 尝试直接找到所有数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if len(numbers) >= 7:
        try:
            return np.array([float(numbers[i]) for i in range(7)])
        except:
            pass
    
    # 如果都无法解析，返回零动作
    logger.warning(f"Could not parse action from text: {text[:100]}...")
    return np.zeros(7)


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    is_healthy = model is not None
    
    response = {
        "is_healthy": is_healthy,
        "status": "running" if is_healthy else "model_not_loaded"
    }
    
    if is_healthy and torch.cuda.is_available():
        response["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1e9
        response["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return jsonify(response)


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_name": getattr(model.config, "_name_or_path", "unknown"),
        "action_dim": 7,
        "action_bins": 256,
        "device": str(device),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    预测端点
    
    请求格式:
    {
        "image": "<base64编码的图像>",
        "instruction": "pick up the red block",
        "temperature": 0.0  (可选)
    }
    
    响应格式:
    {
        "action": [7个浮点数],
        "action_type": "delta_ee"
    }
    """
    import time
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # 解析输入
        image_data = data.get("image")
        instruction = data.get("instruction", "")
        temperature = data.get("temperature", 0.0)
        
        if not image_data:
            return jsonify({"error": "Missing image"}), 400
        if not instruction:
            return jsonify({"error": "Missing instruction"}), 400
        
        # 解码图像
        image = decode_image(image_data)
        
        # 预测
        action = predict_action(
            image,
            instruction,
            temperature=temperature
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        response = {
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "action_type": "delta_ee",
            "inference_time_ms": inference_time
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测端点"""
    try:
        data = request.get_json()
        requests_data = data.get("requests", [])
        
        results = []
        for req in requests_data:
            image = decode_image(req.get("image"))
            instruction = req.get("instruction", "")
            
            action = predict_action(image, instruction)
            results.append({
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
                "action_type": "delta_ee"
            })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="VLA Model Server")
    parser.add_argument(
        "--model", 
        type=str, 
        default="openvla/openvla-7b",
        help="Model name or path"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int4", "int8", "none"],
        default="none",
        help="Quantization option"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    quantization = None if args.quantization == "none" else args.quantization
    load_model(
        model_name=args.model,
        quantization=quantization
    )
    
    # 启动服务器
    logger.info(f"Starting server at {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
