def read_image_paths(file_path):
    """
    读取图像路径文件，支持多种格式：
    1. Python列表字符串格式：['path1', 'path2', ...]
    2. 每行一个路径的格式
    3. JSON格式
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        # 尝试方法1：Python列表格式
        if content.startswith('[') and content.endswith(']'):
            import ast
            try:
                paths = ast.literal_eval(content)
                print(f"✓ 读取为Python列表格式，共 {len(paths)} 个路径")
                return paths
            except:
                pass

        # 尝试方法2：每行一个路径
        lines = content.split('\n')
        paths = []
        for line in lines:
            line = line.strip()
            # 清理行内容
            if line:
                # 移除可能的引号和逗号
                line = line.strip("'\"[] ,;")
                if line and not line.isspace():
                    paths.append(line)

        print(f"✓ 读取为行格式，共 {len(paths)} 个路径")
        return paths

    except Exception as e:
        print(f"读取文件失败: {e}")
        return []


# 使用示例
image_paths = read_image_paths('image_paths.txt')
print(image_paths[1255])
print(f"\n路径示例:")
for i, path in enumerate(image_paths[:5]):
    print(f"  {i + 1}. {path}")
print(f"... 还有 {len(image_paths) - 5} 个路径")