# IsoNet2 使用教程（完整中文翻译版）

> 本文是对 IsoNet2 官方文档的完整中文翻译与整理，力求在**不删减关键信息**的前提下，保持结构清晰、术语准确、表达地道。你可以直接把它当作 IsoNet2 的中文说明书使用。

---

## 0. 简介

**IsoNet2** 是一款用于冷冻电子断层扫描（cryo-ET）重建的深度学习软件包，可以在同一个神经网络中**同时完成缺楔校正、去噪以及 CTF 校正**。  
网络直接从原始断层图（tomogram）中学习信息。与 IsoNet1 相比，IsoNet2 通常能够在约 **十分之一的时间** 内生成**分辨率更高、噪声更低**的 tomogram。

IsoNet2 的输入可以是：

- 完整 tomogram，或者  
- 按 movie frame / 倾转角（tilt）拆分得到的 **even / odd half tomogram**。  

用于 Noise2Noise 训练的 tomogram 对，可以按帧（frame）或按倾转角（tilt）来分成 EVN/ODD 两组。

IsoNet2 主要包含六个模块：

1. `prepare_star`  
2. `deconv`  
3. `make_mask`  
4. `denoise`  
5. `refine`  
6. `predict`  

所有 IsoNet2 命令都基于一个 `.star` 文本文件工作，该文件记录了 tomogram 的路径和相关参数。  
你既可以使用 **图形界面（GUI）**，也可以通过 **命令行（CLI）** 来调用这些模块。

---

## 1. 安装与系统需求

以下内容假定你**完全没有**使用过 Anaconda 或 Linux 环境，所以步骤会写得比较细。

### 1.1 软件需求

Linux 版 IsoNet2 需要：

- CUDA 版本 **≥ 11.8**  
- Conda（任意一种：**Anaconda / Miniconda / Miniforge**）

### 1.2 硬件需求

- Nvidia GTX 1080Ti 或更新型号（更高端显卡同样适用）  
- 显存至少 **10 GB**（越大越好）  

> 一般情况下，24 GB 显存可以比较轻松地使用中等或偏大的网络结构；10 GB 也可以使用，但需要适当减小 cube size 和 batch size。

---

### 1.3 安装 Conda

1. 在浏览器中下载任意一种：  
   - **Miniconda**（体积小，新手推荐）  
   - 或 **Anaconda Distribution**  
   - 或 **Miniforge（conda-forge 社区版）**  

2. 根据官方文档，校验安装包的 **哈希值（hash）**，以确保文件未损坏或被篡改。  

3. 在终端中执行安装脚本（以 Miniconda 为例）：  

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

4. 根据提示一路回车即可。如果某些选项不确定，可以暂时接受默认设置，之后仍可修改。  

5. 安装结束后，关闭当前终端并重新打开一个新的终端。  

6. 在新终端中输入：  

   ```bash
   conda list
   ```  

   如果能看到已安装包列表，说明 Conda 安装成功。

---

### 1.4 安装 CUDA

1. 在 Linux 下查看你的显卡型号，例如：  

   ```bash
   nvidia-smi
   ```  

   或者使用系统设置。  

2. 在 NVIDIA 官网的 “CUDA GPU Compute Capability” 页面中确认你的 GPU **Compute Capability ≥ 3.5**。若低于 3.5，则无法运行 IsoNet2。  

3. 打开 NVIDIA 文档中关于驱动与 CUDA 的对照表，确认当前驱动版本是否支持 `CUDA ≥ 11.8`（一般要求驱动版本 ≥ 520.61.05）。  

4. 从 NVIDIA 驱动下载页面获取与你 GPU 匹配的驱动并安装。  

5. 从 CUDA Toolkit Archive 选择一个与你驱动兼容、且版本 ≥ 11.8 的 **CUDA Toolkit**，按文档步骤安装。  

安装完成后，再次运行：  

```bash
nvidia-smi
```  

你会看到类似的输出：  

```text
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
...
```  

确认 CUDA Version 和 Driver Version 都识别正常。

---

### 1.5 安装 IsoNet2

#### 1.5.1 GUI 安装方式

1. 打开 IsoNet2 的 Github 仓库，在右侧 **Releases** 区下载最新发行版压缩包。  
2. 解压到你希望安装 IsoNet2 的目录。  
3. 发行版中通常已经包含 **GUI 可执行文件**，无需额外编译。  

#### 1.5.2 仅命令行（非 GUI）安装方式

如果只想使用命令行接口（CLI），可以在目标目录下执行：  

```bash
git clone https://github.com/procyontao/IsoNet2.git
```  

这种方式不会包含编译好的 GUI 可执行文件，但更轻量。

#### 1.5.3 创建环境并配置命令

进入 IsoNet2 安装目录：  

```bash
cd IsoNet2
bash install.sh
```  

该脚本会完成：

- 使用 `isonet2_environment.yml` 创建一个 Conda 环境；  
- 自动安装 IsoNet2 所需依赖；  
- 执行 `source isonet2.bashrc`，更新环境变量，保证你能直接在任意目录运行 `isonet.py`。  

你可以在 `~/.bashrc` 的结尾手动加上：  

```bash
source /你的/IsoNet2/路径/isonet2.bashrc
```  

这样每次打开终端时，IsoNet2 的环境都会自动激活并配置好。

安装完成后运行：  

```bash
isonet.py --help
```  

若能看到 IsoNet2 的详细帮助信息，就说明安装成功。

---

## 2. 教程（Tutorial）：从示例数据开始

本教程使用 **EMPIAR-10164** 中的 5 个不成熟 HIV‑1 dMACANC VLP tomogram 作为示例，这些 tomogram 采用基于帧（frame）拆分的 **EVN/ODD 半数据**。

我们将依次演示：

1. 使用 **GUI** 完成完整工作流；  
2. 使用 **命令行（CLI）** 完成相同工作流；  
3. 在教程过程中解释重要参数的含义和常见选择。  

> 对每个模块所有参数的更详细解释，请参考第 3 节 **“IsoNet2 模块说明”**。

---

### 2.0 下载示例 tomogram

在一个新的工作目录中新建文件夹，例如：  

```bash
mkdir isonet2_hiv_tutorial
cd isonet2_hiv_tutorial
```  

从给定的 Google Drive 链接下载 `tomograms_split` 文件夹，并放到当前目录。

你的目录结构应该类似：  

```text
tomograms_split/
├── EVN
│   ├── TS_01_EVN.mrc
│   ├── TS_03_EVN.mrc
│   ├── TS_43_EVN.mrc
│   ├── TS_45_EVN.mrc
│   └── TS_54_EVN.mrc
└── ODD
    ├── TS_01_ODD.mrc
    ├── TS_03_ODD.mrc
    ├── TS_43_ODD.mrc
    ├── TS_45_ODD.mrc
    └── TS_54_ODD.mrc
```  

可以通过：  

```bash
ls -1 tomograms_split/EVN
ls -1 tomograms_split/ODD
```  

确认文件列表无误。

---

## 2.1 GUI 工作流

IsoNet2 的 GUI 提供：

- 数据集与 star 文件管理  
- 参数设置与保存  
- 任务队列与实时日志输出  
- 网络训练过程中的在线预览  

整体流程为：

1. `Prepare`：生成 `tomograms.star`；  
2. （可选）`Denoise` 或 `Deconvolve`：提升对比度，为掩膜准备更干净的数据；  
3. `Create Mask`：生成掩膜；  
4. `Refine`：主训练；  
5. `Predict`：用训练好的模型对 tomogram 生成最终结果。  

---

### 2.1.0 启动 GUI

在终端中输入：  

```bash
IsoNet2
```  

即可启动图形界面。  

如果系统提示与 **SUID sandbox** 相关的错误（常见于某些服务器或无完整桌面环境的机器），可以使用：  

```bash
IsoNet2 -no-sandbox
```  

启动后，请先进入 **Settings** 标签页：

- 选择正确的 **Conda 环境**（即安装 IsoNet2 的那个环境）；  
- 指定 **IsoNet 安装路径**。  

这些设置后续会自动记住。

---

### 2.1.1 Prepare Star

在 GUI 左侧选择 **Prepare** 标签页。

1. 若使用 even/odd 输入，勾选 **Even/Odd Input**；  
2. 在界面中分别指定：  
   - `even` 文件夹 → `tomograms_split/EVN`  
   - `odd` 文件夹 → `tomograms_split/ODD`  

在本教程的 HIV 示例数据中：

- 原始数据中缺少像素尺寸信息，因此需要手动设置 **pixel size = 5.4 Å**；  
- 倾转角范围为 ±60°，因此：  
  - `tilt min = -60`  
  - `tilt max = 60`  

其余与显微镜相关的参数（如 Cs、电压、振幅对比度等），如果你没有更精确的信息，可以暂时使用默认值。

#### create_average 选项

勾选 **create_average** 会令 IsoNet2：  

- 先将每对 even / odd tomogram 进行平均；  
- 生成一个噪声更低的“平均 tomogram”；  
- 该平均 tomogram 非常适合作为后续 **Deconvolve** 或 **Create Mask** 的输入。  

#### Show command

点击 **Show command** 按钮，可以看到当前设置对应的命令行 `isonet.py prepare_star ...`。  
如果你之后更倾向于直接用命令行，可以把这里的命令复制出来保存。

确认参数无误后，点击 **Run**。完成后 GUI 会显示生成的 `tomograms.star`。

如果你已经有一个现成的 star 文件（例如从 RELION5 tomographic pipeline 得到），也可以用 **Load from Star** 直接载入。载入后，你可以在 GUI 中进一步编辑诸如离焦等参数。

> 建议在 `tomograms.star` 中根据 0° tilt 时的 defocus 估值，手动修改 **rlnDefocus** 一列，使每个 tomogram 拥有合理的 defocus 值。

---

### 2.1.2 掩膜前的预处理

为了得到更可靠的掩膜，我们通常希望 tomogram 有更好的对比度。  
在 IsoNet2 中，主要有两条路线：

1. **Denoise 模块**：基于网络的 Noise2Noise 去噪 + CTF 校正（推荐）；  
2. **Deconvolve 模块**：经典的 CTF 反卷积（速度快，掩膜质量通常略逊）。  

#### 2.1.2.1a 使用 Denoise（推荐流程）

1. 打开 **Denoise** 标签页；  
2. 选择合适的网络结构（例如 `unet-medium`）、`subtomo_size`（例如 96）；  
3. 将：  
   - `epochs` 设为 ~20（用于生成掩膜已经足够）；  
   - `CTF_mode` 设为 `network`；  
4. 根据机器实际情况设置 `gpuID`，如 `"0"` 或 `"0,1"`；  
5. `bfactor` 和 `clip_first_peak_mode` 暂时使用默认值；  
6. 建议勾选 **with preview**，以便在训练过程中看到预测切片。  

随后点击 **Submit (In Queue)** 提交任务。  
你可以在任务列表中查看训练过程中的 loss 和日志：

- 日志：`./denoise/jobID/log.txt`  
- 完整损失曲线图：`./denoise/jobID_denoise/loss_full.png`  

训练结束后，你可以进入 **Predict** 标签页，用训练好的模型对 tomogram 进行去噪 + CTF 校正（见下小节）。

#### 2.1.2.1b Predict for Denoise（对去噪网络进行预测）

1. 打开 **Predict** 标签页；  
2. 设置 `gpuID`；  
3. `tomo index` 可设为：  
   - `all`：对 star 文件中所有 tomogram 预测；  
   - 或某个具体编号，如 `1,3,5`；  
4. 从 `./denoise/jobID_denoise` 中选择 checkpoint，例如：  
   ```text
   network_n2n_unet-medium_96_full.pt
   ```  
5. 点击 **Submit** 提交任务；  
6. 任务日志在：`./predict/jobID/log.txt`。  

预测完成后，经 CTF 校正与去噪的 tomogram 将出现在：  

```text
./predict/jobID_predict
```  

这些 tomogram 就可以作为生成掩膜的输入。

---

#### 2.1.2.2 使用 Deconvolve（可选）

若你暂时**不能使用 Denoise 模块的网络 CTF 校正功能**（例如 GPU 资源有限或不方便长时间训练），可以改用 **Deconvolve** 模块来做：

- CTF 反卷积；  
- 提升对比度；  
- 为下一步 `Create Mask` 提供相对干净的 tomogram。  

这一流程与 IsoNet1 的“预处理 + 掩膜生成”思路类似，速度较快但最终掩膜质量稍弱一些。

---

### 2.1.3 Create Mask（生成掩膜）

掩膜是后续训练的关键，它能帮助网络将注意力集中在有样品的区域，避免浪费大量计算在空区域。  
IsoNet2 会根据 tomogram 的局部均值和局部标准差自动生成掩膜。

在 GUI 中打开 **Create Mask** 标签页，选择 **Input Column**：

1. 若使用 ***Denoise*** 结果 → 选 **rlnDenoisedTomoName**；  
2. 若使用 ***Deconvolve*** 结果 → 选 **rlnDeconvTomoName**；  
3. 若在已经 refine 过的 tomogram 上再次 refine → 选 **rlnCorrectedTomoName**；  
4. 若只做了 average 而没有上述步骤（不推荐） → 选 **rlnTomoName**。  

点击 **Submit** 后，IsoNet2 会：

- 从对应列读取 tomogram；  
- 计算局部统计；  
- 输出掩膜到 `./make_mask/jobID/`，并在 `tomograms.star` 中填写好 **rlnMaskName**。  

任务日志在：  

```text
./make_mask/jobID/log.txt
```  

---

### 2.1.4 Refine（主训练：缺楔校正 + 去噪）

Refine 是 IsoNet2 的核心训练模块，用于：

- 在 Noise2Noise 设定下进行去噪；  
- 同时显式学习缺楔校正；  
- 可以额外包含 CTF 相关的建模（`CTF_mode` 选择）。  

在 GUI 中打开 **Refine** 标签页：

1. 勾选 **Even/Odd Input**（针对 Noise2Noise 训练）；  
2. 指定 `gpuID`，如 `"0"` 或 `"0,1,2,3"`；  
3. 根据显存选择网络大小与 `subtomo_size`（例如 `unet-medium` + cube = 96）；  
4. 设置 `mw_weight`，控制缺楔损失的权重（如 200）；  
5. 选择 `CTF_mode`（例如 `network`）；  
6. `bfactor`、`epochs` 等参数按需要设定：  
   - 对最终结果的训练，一般建议 `epochs ≥ 50`。  

点击 **Submit (In Queue)** 提交任务。  
在任务列表中点击可以查看训练日志与预览结果：

- 日志：`./refine/jobID/log.txt`  
- 完整 loss 曲线：`./refine/jobID_refine/loss_full.png`  

你也可以多次提交不同参数的 refine 任务（例如不同 CTF_mode 或不同 mw_weight），并在 **Jobs Viewer** 中统一查看和管理。

> 注意：  
> - **Submit (In Queue)** 会通过任务队列依次运行多个训练任务；  
> - **Submit (Run Immediately)** 会直接启动，不经过队列管理，如连续点多次容易造成多个训练任务同时抢占 GPU。

---

### 2.1.5 Predict for Refine（对最终模型预测）

Refine 训练完成后，在 **Predict** 标签页中：

1. 选择对应的 `gpuID`；  
2. `tomo index` 设为 `all` 或某些具体 index；  
3. 从 `./refine/jobID_refine` 目录中选取 checkpoint，例如：  
   ```text
   network_isonet2-n2n_unet-medium_96_full.pt
   ```  
4. 点击 **Submit** 提交；  
5. 日志：`./predict/jobID/log.txt`。  

预测完成后，经 **CTF 校正 + 去噪 + 缺楔补偿** 的 tomogram 会存放在：  

```text
./predict/jobID_predict
```  

这就是 IsoNet2 的最终输出，可作为后续 STA、分割或可视化分析的输入。

---

## 2.2 命令行（CLI）工作流

当你对 GUI 流程比较熟悉后，很可能更习惯使用命令行来：

- 批量处理多个数据集；  
- 轻松调整超参数并记录命令；  
- 写脚本整合到你现有的处理管线。  

你也可以在 GUI 中点击 **Show command**，把对应命令复制出来，在终端中运行。

下面给出与上文 GUI 教程对应的命令行示例。

---

### 2.2.1 prepare_star

示例命令（使用 even/odd half tomogram）：  

```bash
isonet.py prepare_star \
  --even tomograms_split/EVN \
  --odd tomograms_split/ODD \
  --create_average True \
  --pixel_size 5.4 \
  --defocus "[39057,14817,25241,29776,15463]"
```  

说明：

- `--pixel_size 5.4`：由于示例数据不含像素尺寸信息，这里手动指定为 5.4 Å；  
- `--defocus`：给出每个 tomogram 对应的 defocus（以 Å 为单位）；  
- `--create_average True`：生成平均 tomogram。  

你也可以将 `defocus` 先设置为 `None`，之后编辑 `tomograms.star`：  

```bash
nano tomograms.star   # 或用 vim、gedit 等编辑器
```  

手动修改 `_rlnDefocus` 列。

---

### 2.2.2.1a denoise（可选但推荐）

利用 Noise2Noise 去噪 + 网络式 CTF 校正：  

```bash
isonet.py denoise tomograms.star \
  --with_mask True \
  --gpuID 0 \
  --CTF_mode network
```  

训练完成后，用模型对原始 tomogram 进行预测：  

```bash
isonet.py predict tomograms.star \
  denoise/network_n2n_unet-medium_96_full.pt \
  --gpuID 0
```  

预测结果会保存为去噪 + CTF 校正的 tomogram。

---

### 2.2.2.2 deconv（可选）

使用经典 CTF 反卷积增强对比度：  

```bash
isonet.py deconv tomograms.star --snrfalloff 0.7
```  

你可以根据数据情况调整 `snrfalloff`（通常在 0–1 之间）。

---

### 2.2.3 make_mask

基于统计特征自动生成掩膜：  

```bash
isonet.py make_mask tomograms.star --input_column rlnDenoisedTomoName
```  

`input_column` 的选择规则：

- 若已进行 Denoise → `rlnDenoisedTomoName`；  
- 若使用 Deconvolve → `rlnDeconvTomoName`；  
- 若在 refine 后再次 refine → `rlnCorrectedTomoName`；  
- 若只用 average（不推荐） → `rlnTomoName`。  

---

### 2.2.4 refine

训练 IsoNet 模型进行缺楔补偿和去噪：  

```bash
isonet.py refine tomograms.star \
  --with_mask True \
  --gpuID 0 \
  --mw_weight 200 \
  --bfactor 200 \
  --CTF_mode network
```  

说明：  

- `--with_mask True`：使用之前生成的掩膜；  
- `--mw_weight 200`：对缺楔区域的损失加大权重，更强调缺楔补偿；  
- `--bfactor 200`：适合非细胞样品，可加强高频。对于细胞 tomogram 通常推荐设为 0。  

---

### 2.2.5 predict

使用训练好的 refine 模型生成最终 tomogram：  

```bash
isonet.py predict tomograms.star \
  isonet_maps/network_isonet2-n2n_unet-medium_96_full.pt \
  --gpuID 0
```  

输出会写入新的 MRC 文件，并在 star 中记录到相应列（如 `rlnCorrectedTomoName`）。

---

## 3. IsoNet2 模块说明（详细）

下面对各个模块的主要用途与重要参数做相对完整的说明，方便你在真实数据上做更灵活的调整。

---

### 3.1 模块：prepare_star

**功能：**

- 生成 `tomograms.star` 文件。  
- 兼容 RELION5 tomographic pipeline 的 star 格式。  
- 支持以下输入：  
  - 完整 tomogram；  
  - even / odd half tomogram。  

**典型用途：**

- 记录每个 tomogram 的路径、像素尺寸、倾转范围、显微镜参数等；  
- 记录每个 tomogram 需要抽取的 subtomogram 数量；  
- 为之后的 `denoise / deconv / make_mask / refine / predict` 提供统一的元数据入口。  

**重要参数（仅列出常用）：**

- `full`：完整 tomogram 所在目录。  
- `even`：even half tomogram 所在目录。  
- `odd`：odd half tomogram 所在目录。  
- `create_average`：是否生成 even+odd 的平均 tomogram。  
- `number_subtomos`：每个 tomogram 要提取的 subtomogram 数量（越大相当于训练曝光越多）。  
- `pixel_size`：像素尺寸（Å）。若 tomogram 文件内部没有该信息，则必须在此提供。  
- `cs`：球差（mm）。  
- `voltage`：加速电压（kV）。  
- `ac`：振幅对比度。  
- `defocus`：离焦值（可为一个统一值，或多个值组成的列表）。  
- `tilt_min` / `tilt_max`：倾转角范围（°）。  
- `mask_folder`：若已存在掩膜，可在此指定目录，IsoNet2 会写入 `rlnMaskName`。  
- `coordinate_folder`：若已有坐标文件，可以从中推导 subtomogram 数量。  
- `star_name`：输出的 star 文件名称。  

---

### 3.2 模块：denoise

**功能：**

- 用于 Noise2Noise 去噪；  
- 可选地包含 CTF 相关建模（`CTF_mode`）；  
- 训练速度相对 refine 更快，适合初步测试和生成掩膜前的预处理。  

**重要参数：**

- `arch`：网络结构，例如 `unet-small`、`unet-medium`、`unet-large`。  
- `batch_size`：每个 batch 的 subtomogram 数；若留空，默认 `2 × (GPU 数)`。  
- `cube_size` / `subtomo_size`：训练 subtomogram 的立方体尺寸（像素），需为 16 的倍数且 ≥ 64。  
- `bfactor`：频域上用于增强高频的 B 因子；  
  - 细胞 tomogram 推荐 0；  
  - 孤立颗粒可以设为 200–300。  
- `CTF_mode`：  
  - `None`：不做 CTF 校正；  
  - `phase_only`：相位翻转；  
  - `network`：对输入施加 CTF 形状滤波；  
  - `wiener`：使用维纳滤波器对 target 做 CTF 反卷积。  
- `epochs`：训练总轮数。  
- `learning_rate`、`learning_rate_min`：学习率及其下限。  
- `gpuID`：逗号分隔的 GPU ID。  
- `mixed_precision`：是否启用 float16 混合精度。  
- `loss_func`：损失函数（L2 / Huber / L1）。  
- `with_preview`：若为 True，训练完成后会执行一次预测并生成预览 tomogram。  
- `snrfalloff`、`deconvstrength`、`highpassnyquist`：与 CTF 及反卷积相关的参数。  

---

### 3.3 模块：deconv

**功能：**

- 用于对 tomogram 做 CTF 反卷积，从而增强对比度；  
- 特别适合：  
  - 非相位板数据；  
  - GPU 不足以跑完整网络 CTF 校正时；  
  - 仅仅需要一个较好掩膜时的预处理。  

**重要参数：**

- `star_file`：输入 STAR。  
- `input_column`：读取 tomogram 的列（默认为 `rlnTomoName`，也可指定为 average 列）。  
- `output_dir`：反卷积后 tomogram 的输出目录。  
- `snrfalloff`：控制随频率变化的 SNR 衰减（默认 1.0，常用范围 0–1）。  
- `deconvstrength`：整体反卷积强度（默认 1.0，可调到 2–5 观察效果）。  
- `highpassnyquist`：高通滤波的 Nyquist 比例（默认 0.02，用于去除极低频的背景梯度）。  
- `chunk_size`：若内存有限，可指定一个块大小，使 tomogram 分块处理。  
- `overlap_rate`：块之间的重叠比例（默认 0.25，常用 0.25–0.5）。  
- `ncpus`：CPU worker 数。  
- `phaseflipped`：若输入数据已经做过相位翻转，则设为 True。  
- `tomo_idx`：只处理指定 index 的 tomogram。  

---

### 3.4 模块：make_mask

**功能：**

- 根据 tomogram 的局部平均密度和局部标准差生成掩膜；  
- 掩膜会写入 star 文件并用于 refine 中的采样。  

**重要参数：**

- `star_file`：输入 STAR。  
- `input_column`：用于生成掩膜的 tomogram 列，如 `rlnDeconvTomoName` 或 `rlnDenoisedTomoName`。  
- `output_dir`：掩膜输出目录。  
- `patch_size`：局部统计的窗口大小（默认 4）。  
- `density_percentage`：按局部密度百分位数保留的体素比例（默认 50）。  
- `std_percentage`：按局部标准差百分位数保留的体素比例（默认 50）。  
- `z_crop`：在 z 方向裁掉的上下比例（默认 0.2，即各裁 10%）。  
- `tomo_idx`：仅对指定的 tomogram 生成掩膜。  

---

### 3.5 模块：refine

**功能：**

- IsoNet2 的主训练模块；  
- 支持两种模式：  
  1. `isonet2`：使用完整 tomogram；  
  2. `isonet2-n2n`：使用 even/odd half tomogram 做 Noise2Noise 训练。  

**在 denoise 的基础上，refine 额外关心：**

- 缺楔补偿；  
- 掩膜策略；  
- 多种噪声建模与数据增强。  

**关键参数（除 denoise 中已有的外，新增/重要部分）：**

- `method`：  
  - `isonet2`：单图模式；  
  - `isonet2-n2n`：Noise2Noise 模式（自动检测 even/odd）。  
- `mw_weight`：缺楔损失权重；  
  - 0：关闭缺楔 masked loss，仅做去噪；  
  - 20–200：常见设置，偏重缺楔区域重建。  
- `noise_level` / `noise_mode`：在训练中注入额外噪声以及相应滤波模式。  
- `apply_mw_x1`：是否对输入 subtomogram 施加缺楔掩膜。  
- `clip_first_peak_mode`：控制 CTF 第一峰的截断方式：  
  - 0：不处理；  
  - 1：常数截断；  
  - 2：negative sine；  
  - 3：cosine。  
- `num_mask_updates`：训练过程中掩膜更新次数。  
- `random_rot_weight`：随机旋转增强的权重。  
- `with_deconv`：是否在 refine 过程中自动结合 deconv。  
- `with_mask`：是否启用掩膜（一般强烈建议 True）。  

---

### 3.6 模块：predict

**功能：**

- 使用训练好的模型（来自 denoise 或 refine）对 tomogram 执行前向预测；  
- 输出去噪或缺楔补偿后的 tomogram。  

**重要参数：**

- `star_file`：输入 STAR。  
- `model`：训练得到的 `.pt` checkpoint 路径。  
- `output_dir`：预测 tomogram 的输出目录。  
- `gpuID`：使用的 GPU ID。  
- `input_column`：用于预测的 tomogram 列，如 `rlnDeconvTomoName` 等。  
- `apply_mw_x1`：预测时是否对输入块应用缺楔掩膜。  
- `isCTFflipped`：声明输入 tomogram 是否已经做过相位翻转。  
- `padding_factor`：块拼接时的 padding 范围（默认 1.5）。  
- `tomo_idx`：只预测指定 index 的 tomogram。  
- `output_prefix`：输出 MRC 文件名的前缀。  

---

## 4. 常见问题（FAQ）

### Q1：什么时候使用 even/odd 半 tomogram？什么时候用完整 tomogram？

- 若你的数据可以从 movie / tilt-series 拆分为 **even/odd**，强烈建议用 **Noise2Noise（isonet2-n2n）** 工作流，效果更佳；  
- 若你只有最终重建 tomogram，无法拆分，则使用 **isonet2**（单图模式）。

---

### Q2：每个 tomogram 应该抽取多少个 subtomogram？需要训练多少 epoch？

- IsoNet2 会在训练中**动态抽取** subtomogram（不像 IsoNet1 需要预先抽取）；  
- 增加 `number_subtomos` 相当于增加训练数据量；  
- 学术实测中，一般**每个 epoch 抽取 ~3000 个 subtomogram** 作为默认设定；  
- 通常不建议总训练 epoch 少于 **50**，除非你只是做快速测试。  

---

### Q3：训练时显存不够怎么办？

可以尝试：

1. 开启 `mixed_precision`；  
2. 调小 `batch_size`（下限为 GPU 数）；  
3. 使用较小的 `arch`（如 `unet-small`）；  
4. 减小 `cube_size`；  
5. 在 `deconv` / `predict` 等步骤使用 `chunk_size` + `overlap_rate` 分块处理大 tomogram。  

---

### Q4：什么时候应该用 deconv 模块？

以下情况适合使用 `deconv`：

- 只有最终重建 tomogram，没有 half tomogram，想要做缺楔校正（类似 IsoNet1 流程）；  
- 有 half tomogram，但你只想快速获得一个还不错的掩膜：  
  1. 在 prepare_star 中启用 `create_average`；  
  2. 用 deconv 对平均 tomogram 做反卷积；  
  3. 用 deconv 结果作为 make_mask 的输入。  

---

### Q5：掩膜在 refine 中有多重要？

非常重要。掩膜可以：

- 提高训练样本利用率（更少采到纯空背景）；  
- 让网络集中学习样品区域的细节；  
- 提高整体训练稳定性。  

一般我们**强烈推荐在 refine 时始终使用掩膜**。  
对于单纯做去噪的 `denoise`，掩膜则不是必须的。

---

### Q6：如果自动掩膜漏掉了样品区域怎么办？

你可以：

1. 放宽掩膜参数：增大 `density_percentage` 或 `std_percentage`；  
2. 编辑 star 文件中 `_rlnMaskBoundary` 相关参数；  
3. 自己制作掩膜并通过 `mask_folder` 或直接编辑 `rlnMaskName` 指定。  

---

### Q7：refine 中 CTF_mode 选 network 还是 wiener 更好？

- `CTF_mode = network`，通常搭配 `clip_first_peak_mode = 1` 是一种比较“稳妥”的设置，高分辨率表现较好；  
- `clip_first_peak_mode` 改成 2 或 3，有时能得到更好的低频对比度（要结合数据查看）；  
- `CTF_mode = wiener` 在调参得当时也很强，但对 `snrfalloff` 与 `deconvstrength` 比较敏感，需要多试几组参数。  

总的来说：

- **network 模式更偏向开箱即用**；  
- **wiener 模式需要更多经验和调参时间**。  

---

### Q8：mw_weight 应该设多大？

一般用于控制“缺楔区域的损失权重”，常见策略：

- **20–200**：常用范围，会显著增强缺楔补偿效果；  
- **0**：相当于关闭缺楔 masked loss，此时网络更多在做“全局去噪 + 轻微补偿”，而不是专门针对缺楔。  

可以先从 100 左右开始尝试，根据结果再微调。

---

（完）
