#!/usr/bin/env python3
import os
from pathlib import Path

def generate_code_content_report(ros2_ws_path, output_path):
    """生成包含完整代码内容的报告"""
    # 定义需要包含的核心文件类型
    include_patterns = [
        "*.py", "*.msg", "*.sh", "*.md", 
        "CMakeLists.txt", "package.xml", ".envrc"
    ]
    
    # 定义需要排除的目录
    exclude_dirs = {
        "test", "__pycache__", "resource", "install", "build", "log"
    }
    
    # 定义特殊文件处理（只包含关键内容）
    special_files = {
        "README.md": "只提取核心描述部分",
        "LICENSE": "只保留许可证类型"
    }
    
    report = []
    ros2_ws_path = Path(ros2_ws_path)
    file_count = 0
    total_size = 0
    
    # 遍历目录结构
    for root, dirs, files in os.walk(ros2_ws_path):
        # 排除不需要的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # 处理文件
        for file in files:
            file_path = Path(root) / file
            
            # 检查是否为需要包含的文件类型
            if any(file_path.match(pattern) for pattern in include_patterns):
                relative_path = file_path.relative_to(ros2_ws_path)
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 处理特殊文件
                    if file in special_files:
                        content = process_special_file(file, content)
                    
                    # 添加到报告
                    report.append({
                        "path": str(relative_path),
                        "content": content
                    })
                    
                    file_count += 1
                    total_size += len(content)
                    
                except Exception as e:
                    print(f"警告: 无法读取文件 {relative_path} - {str(e)}")
    
    # 生成报告文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ROS2 工作空间核心代码内容报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"工作空间位置: {ros2_ws_path.resolve()}\n")
        f.write(f"包含文件数: {file_count}\n")
        f.write(f"总代码量: {total_size//1024} KB\n\n")
        
        # 添加目录结构概览
        f.write("目录结构概览:\n")
        f.write("-" * 80 + "\n")
        f.write(get_directory_overview(ros2_ws_path, exclude_dirs))
        f.write("\n\n")
        
        # 添加每个文件的完整内容
        for item in report:
            f.write("=" * 80 + "\n")
            f.write(f"文件路径: {item['path']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(item['content'])
            f.write("\n\n")  # 添加额外空行分隔文件
        
        # 添加总结
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    return output_path

def process_special_file(filename, content):
    """处理特殊文件，提取关键内容"""
    if filename == "LICENSE":
        # 提取许可证类型
        if "Apache License" in content:
            return "许可证类型: Apache-2.0"
        elif "MIT License" in content:
            return "许可证类型: MIT"
        return "许可证类型: 其他"
    
    elif filename == "README.md":
        # 只保留核心描述部分
        lines = content.split('\n')
        core_lines = []
        for line in lines:
            if line.startswith(("#", "##", "###", "**")):
                core_lines.append(line)
        return "\n".join(core_lines)
    
    return content

def get_directory_overview(root_path, exclude_dirs):
    """生成目录结构概览"""
    overview = []
    root_path = Path(root_path)
    
    for path in sorted(root_path.rglob('*')):
        if path.is_dir() and path.name in exclude_dirs:
            continue
            
        relative_path = path.relative_to(root_path)
        indent = "  " * (len(relative_path.parts) - 1)
        
        if path.is_dir():
            overview.append(f"{indent}└── {path.name}/")
        else:
            # 只显示文件类型
            if any(path.match(pattern) for pattern in ["*.py", "*.msg", "*.sh"]):
                file_type = {
                    ".py": "Python脚本",
                    ".msg": "ROS消息定义",
                    ".sh": "Shell脚本",
                    ".md": "文档",
                    ".txt": "文本文件"
                }.get(path.suffix, "文件")
                overview.append(f"{indent}    ├── {path.name} ({file_type})")
    
    return "\n".join(overview)

if __name__ == "__main__":
    # 配置路径
    project_root = Path(__file__).parent.parent  # 假设脚本在项目根目录下的scripts文件夹中
    ros2_ws_path = project_root / "ros2_ws"
    output_path = project_root / "full_code_report.txt"
    
    # 生成报告
    report_path = generate_code_content_report(ros2_ws_path, output_path)
    print(f"完整代码报告已生成至: {report_path}")
    print(f"包含所有核心文件的完整代码内容")