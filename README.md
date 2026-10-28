# PPG运动伪影去除 · 软件工作台（app分支）

这个分支放可运行的软件：本地网页工作台，以及打包成单文件exe的两个版本。算法、训练脚本和实验报告在`main`分支，开放数据子集在`datasets`分支，这里只保留跑软件需要的那部分代码。

## 两个版本

`webapp/`是v2，默认版本。内置演示数据按真实PPG形态和物理幅度IMU合成，演示时显示“链路自检”而不是PRD，静息场景会触发旁路。

`webapp_v1/`是原始版，演示数据是纯正弦，演示时显示PRD。它只用来对照，交付建议用v2。两版调用的是同一套模型和接口，差别只在演示数据与演示态展示。

演示数据与真实数据的行为不一样：演示数据的PRD没有物理意义，所以v2在演示态换成链路自检（长度、有限值、心率、伪影占比、是否旁路）；上传真实文件后会自动恢复PRD、Pearson和SNR的显示。

## 目录

```
app/
├── webapp/                    # v2源码：server.py、static/index.html、使用说明书.md、start.sh
│   └── packaging/             # PyInstaller配置与构建脚本
├── webapp_v1/                 # v1源码
├── scripts/                   # 软件依赖：ppg_api、网络结构、pipeline、run_denoise
├── models/ppg_denoise_v1/     # lite / full / no_imu三套权重
├── .github/workflows/         # 构建与发布工作流
├── requirements.txt
└── run_demo.sh
```

## 模型预设

`lite`是5个残差模型加3个频域模型，15MB，PRD46.37，软件默认用它；`full`是41加3个成员，108MB，PRD46.57，用来复现报告里的最终数字；`no_imu`是3个单通道模型，输入没有IMU时自动切过去。权重的sha256写在`manifest.json`里。

## 源码启动

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

OMP_NUM_THREADS=2 bash webapp/start.sh
# 或者：OMP_NUM_THREADS=2 python webapp/server.py --port 8000
```

启动后会自动打开浏览器，也可以手动访问http://127.0.0.1:8000。`OMP_NUM_THREADS=2`别漏，torch默认按核数开线程，128核机器上30s信号会从0.2s拖到5s左右。模型在首次请求时才加载，lite一共8个权重，大约1到3秒。

## 单文件exe

产物有两个，都是约294MB的单文件，无控制台窗口：`PPG-Workbench-v2.exe`（默认）和`PPG-Workbench-v1.exe`（对照）。双击后自动打开浏览器，端口被占会顺延；首次启动要把运行时解压到临时目录，20到60秒属正常；退出点网页右上角的按钮，日志写在同一目录的`workbench.log`里。

构建方式有三种。最省事的是推送到`app`分支或手动触发Actions，跑完在Artifacts里拿两个zip；推`v*`tag会自动建Release并附上两个zip和校验和，tag要打在`app`分支上。Windows本机双击`webapp/packaging/build_windows.bat`即可；Linux和macOS用`bash webapp/packaging/build_linux.sh "v1 v2"`，主要用来验证打包逻辑。PyInstaller不能交叉编译，Windows的exe必须在Windows或Actions上构建。

## 网页功能

输入支持内置演示数据（静息、中等、强运动三档）和外部文件（npz、csv、json）。输出是三波形（原波形、伪影估计、滤净波形，另有一条默认隐藏的物理先验）、逐窗心率曲线和质量指标（伪影占比、是否旁路、SNR、运行模式、耗时）。交互有滚轮缩放、拖拽平移、双击复位、悬停读数、图例开关和明暗主题，最近的处理记录存在浏览器本地。结果可导出成npz、json、csv和png。

## 接口

- `GET /api/health`：服务状态、版本、torch版本、已加载模型
- `GET /api/info`：预设、限制、演示场景
- `GET /api/demo?scenario=rest|moderate|strong`：预览演示数据，不推理
- `POST /api/denoise`：主接口，multipart或json
- `POST /api/selfcheck`：用合成样例跑一遍端到端自检
- `GET /api/runs`、`GET /api/run/<id>`：列出或读取缓存结果
- `GET /api/download/<id>/{npz,json,csv,png}`：下载产物
- `POST /api/shutdown`：退出服务，无控制台版靠它关程序

```bash
curl -X POST http://127.0.0.1:8000/api/denoise -F source=demo -F scenario=moderate -F preset=lite
curl -X POST http://127.0.0.1:8000/api/denoise -F source=file -F file=@my.npz -F preset=lite
```

返回里主要看`result.ppg_clean`、`result.hr_median`、`result.quality`和`series`；输入带参考波形时多一个`reference_metrics`，演示数据多一个`link_check`。

## 输入格式

npz是推荐格式：

```python
np.savez_compressed("my.npz",
    ppg=ppg,          # 必需，(N,)或(n_lead, N)，任意单位
    acc=acc,          # 可选，(3,N)/(N,3)/(N,)
    gyro=gyro,        # 可选，有则用ACC_x+Gyro_y异质两路参考
    reference=clean,  # 可选，干净PPG真值，用来显示PRD/Pearson/SNR
    fs=100.0,         # 可选，缺省时界面左侧要填
)
```

csv用`ppg, acc_x/y/z, gyro_x/y/z, reference`列名，但csv不带采样率，界面左侧要手动填。json就是对应的字典。

约束不多：文件不超过64MB、采样点不超过200万、任意采样率（内部统一重采样到100Hz）、任意长度（超过30s滑窗重叠相加，短于1s报错），NaN会自动插值。

## 打包做了哪些处理

onefile把模型、网页和`pipeline.py`一起打进exe，运行时解压到`sys._MEIPASS`；`console=False`关掉控制台，启动时把`stdout`和`stderr`重定向到日志文件，否则打印会出错；没有Ctrl+C，所以加了`/api/shutdown`和页面上的退出按钮；启动时限制torch线程数；端口从8000起顺延到8019；构建时装CPU版torch，避免把CUDA打进包。更细的说明在`webapp/packaging/README.md`。

## 常见问题

- 双击没反应或等很久：首次启动在解压运行时，20到60秒正常，看同目录的`workbench.log`。
- Windows提示“未知发布者”：程序没签名，点“更多信息→仍要运行”即可。
- 杀毒软件报毒：PyInstaller单文件常见误报，加白名单，或者改用onedir版本。
- 想关掉程序：点页面右上角按钮，或者任务管理器结束进程。
- 端口被占：会自动往后找，也可以用`--port`指定。
- 只有三波形没有PRD：输入里没有`reference`，属正常。
- 提示“信号干净，已透传”：伪影占比低于阈值，旁路生效了。

## 与其它分支的关系

`main`放算法、训练评估脚本、实验报告和权重源，本分支的`models/`与`scripts/`关键文件从那里同步；`datasets`放可再分发的公开数据子集，需要的话`git fetch origin datasets && git checkout origin/datasets -- datasets/`。

## 说明

代码供赛题评审和学术复现使用，数据集不随本分支分发。上游参考实现`uEDA-Net`和`Kernel-Adaptive-Filtering`只在`main`分支的实验里作对照。
