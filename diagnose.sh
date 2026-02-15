#!/bin/bash
# 诊断脚本 - 检查VLA部署状态

echo "=========================================="
echo "VLA部署诊断工具"
echo "=========================================="
echo ""

# 检查本地Isaac Sim
echo "1. 检查本地Isaac Sim..."
if [ -d "/home/vincent/isaac-sim" ]; then
    echo "   ✓ Isaac Sim目录存在"
    cat /home/vincent/isaac-sim/VERSION 2>/dev/null || echo "   ⚠ 无法读取版本"
else
    echo "   ✗ Isaac Sim目录不存在"
fi
echo ""

# 检查Python代码更新
echo "2. 检查本地代码更新..."
cd /home/vincent/Desktop/code/VLA 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   当前分支: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "   最近提交:"
    git log -1 --oneline 2>/dev/null || echo "   ⚠ 无法读取git历史"
    echo ""
    echo "   server_deploy.py修改状态:"
    git diff HEAD server/server_deploy.py | head -20
else
    echo "   ✗ 无法进入项目目录"
fi
echo ""

# 检查远程服务器连接
echo "3. 检查远程VLA服务器..."
echo "   测试health端点:"
curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "   ✗ 无法连接"
echo ""

echo "4. 测试predict端点:"
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "instruction": "test"}' \
  -w "\nHTTP Status: %{http_code}\n" 2>&1 | head -10
echo ""

echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "如果远程服务器返回500错误，请确保:"
echo "1. 远程服务器已执行: git pull"
echo "2. 远程服务器已重启: python server_deploy.py"
echo "3. 检查远程服务器日志查看详细错误"
