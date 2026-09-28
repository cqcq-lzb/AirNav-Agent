# 经支气管肺结节自动导航系统 v3

## 一、系统组成

```text
airway_navigation_system_v3.py
interactive_navigation_stage4_v3.py
run_stage4_server_pipeline.py
server_config.json
check_system_environment_v3.py
启动自动导航系统_v3.bat
```

## 二、完整工作流程

```text
选择胸部 CT
→ SSH 上传服务器
→ 服务器运行气道分割
→ 服务器运行结节分割
→ 生成全部结节候选病例包
→ 自动下载到 Windows
→ 自动解压
→ 打开三方位二维 + 三维地图 + 虚拟支气管镜界面
→ 点击不同结节，本地直接重新规划
```

服务器流水线仍调用现有：

```text
/data2/home/wcq/nnUNet/nav_project/engineering_demo/run_full_pipeline.py
```

客户端会自动把：

```text
run_stage4_server_pipeline.py
```

上传到服务器工程目录，不需要手工上传。

## 三、放置位置

把整个压缩包解压到：

```text
D:\airway_navigation_v1
```

最终应为：

```text
D:\airway_navigation_v1\
├─ airway_navigation_system_v3.py
├─ interactive_navigation_stage4_v3.py
├─ run_stage4_server_pipeline.py
├─ server_config.json
├─ check_system_environment_v3.py
├─ 启动自动导航系统_v3.bat
└─ cases\
```

## 四、运行环境

解释器：

```text
D:\Anaconda\envs\dicom\python.exe
```

检查环境：

```bat
D:\Anaconda\envs\dicom\python.exe ^
D:\airway_navigation_v1\check_system_environment_v3.py
```

需要：

```text
numpy
scipy
scikit-image
SimpleITK
VTK
PySide6
paramiko
```

## 五、启动

双击：

```text
启动自动导航系统_v3.bat
```

或在 PyCharm Terminal 运行：

```bat
D:\Anaconda\envs\dicom\python.exe ^
D:\airway_navigation_v1\airway_navigation_system_v3.py
```

## 六、操作

1. 点击“选择 CT”；
2. 病例编号会根据文件名自动生成，可手工修改；
3. 输入服务器 `wcq` 用户密码；
4. GPU 保持 `5`；
5. 点击“开始服务器分割并打开导航”；
6. 等待上传、分割、打包和下载；
7. 处理完成后自动打开交互导航。

本地结果保存在：

```text
D:\airway_navigation_v1\cases\<病例编号>\stage4_package
```

## 七、导航界面 v3

### 左侧

- 全部结节候选；
- 器械外径和安全余量；
- 推荐路径指标；
- 当前分支、局部气道直径和完整分叉序列。

### 中间

- 大尺寸虚拟支气管镜；
- 屏幕前景完整长箭头；
- 起点、终点、逐点前进、跨 10 点、播放、速度和循环。

### 右侧

- 轴位；
- 冠状位；
- 矢状位；
- 三维气道地图；
- 青色实时位置球和光环；
- 白色实时方向箭头。

## 八、虚拟镜箭头 v3

旧版三维箭头与镜头视线共线时，会产生透视缩短，只能看到尾部。

v3 改成屏幕前景 HUD 长箭头：

```text
直行：箭头完整向上
左转：箭头完整指向左前方
右转：箭头完整指向右前方
```

箭头始终显示完整箭杆和箭头，不再被气道深度透视压缩。

## 九、服务器设置

默认设置位于：

```text
server_config.json
```

默认服务器：

```text
gpu-node
端口 22
用户 wcq
GPU 5
```

默认工程目录：

```text
/data2/home/wcq/nnUNet/nav_project/engineering_demo
```

默认 nnU-Net 环境变量：

```text
nnUNet_raw=/data2/home/wcq/nnUNet/nnUNet_raw
nnUNet_preprocessed=/data2/home/wcq/nnUNet/nnUNet_preprocessed
nnUNet_results=/data2/home/wcq/nnUNet/nnUNet_results
```

## 十、说明

- 服务器负责深度学习分割；
- Windows 负责交互、三维重建、中心线提取和目标切换后的重新规划；
- 切换结节目标时不会再次运行 nnU-Net；
- 虚拟支气管镜是术前 CT 模拟视图；
- 当前没有真实支气管镜视频、器械跟踪或术中配准；
- 本系统为工程研究演示，不用于临床诊断或治疗。
