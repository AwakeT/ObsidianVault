# DINO-GeoSemMap 数据集构成

> 版本：v0.1
> 日期：2026-06-02
> 关联文档：[07 架构设计](07_DINO-GeoSemMap_架构设计.md) ｜ [08 训练计划](08_DINO-GeoSemMap_训练计划.md) ｜ [09 存储规划](09_DINO-GeoSemMap_存储规划.md)

---

## 0. 文档定位

本文回答各 head **"用什么数据训"**——数据源、配比、in-domain 策略、Habitat 渲染规格、关键映射表。配合 [08](08_DINO-GeoSemMap_训练计划.md) 的训练计划与 [09](09_DINO-GeoSemMap_存储规划.md) 的存储规划。

---

## 1. 总约束（决定数据构成的前提）

| 约束 | 内容 | 来源 |
|---|---|---|
| 相机 | 原生 FOV 120°×85°，**输入统一处理为 1280×720 去畸变针孔图** | 部署设定 |
| FOV 裁剪 | **B 方向**：去畸变后裁到较窄 rectilinear（~85–95°），避开广角四角拉伸 | 本文决策 |
| 工作分辨率 | **固定 896×504（16:9，2304 tokens）**，所有输入统一 resize 到此，**不做多分辨率** | 本文决策 |
| 多焦距 | **保留 Canonical Transform**：训练数据多源多焦距，需归一化；焦距增强用 crop→resize 到 896×504 | 本文决策 |
| 数据主力 | B 裁窄后焦距变长、接近公开窄 FOV 数据 → **公开真实数据为主，Habitat 为 in-domain 部分补充** | 本文决策 |
| Habitat | **尚未采集**，从零按部署内参渲染（无历史包袱）；一次渲染同时出四 head 标签 | 本文决策 |
| 每轮全 head | 每一轮都对全部四个 head 做效果确认 | [08 §2.1 铁律](08_DINO-GeoSemMap_训练计划.md) |
| 残余风险 | 无真机数据，sim eval 为 sim→sim；需中期补小真机 eval（见 §5） | 本文 |

---

## 2. 各 head 数据集构成

> 配比为 **pilot 起步提案**，非定论，预验证轮 L2 调。

### 2.1 深度（阶段0）

| 来源                  | 角色                       | 真实/合成 | 提案配比 |
| ------------------- | ------------------------ | ----- | ---- |
| ARKitScenes         | 主力，量大、室内、LiDAR metric    | 真实    | ~35% |
| ScanNet / ScanNet++ | 高质量真实几何                  | 真实    | ~25% |
| Hypersim            | 稠密无洞 + 锐利边界              | 合成    | ~15% |
| NYUv2               | 经典补充（小）                  | 真实    | ~5%  |
| **Habitat**         | in-domain 视角 + 精确 FOV 对齐 | 合成    | ~20% |

- 每条样本需 **metric depth + valid_mask + K**；按 [08 §3.2](08_DINO-GeoSemMap_训练计划.md) Canonical Transform 归一化；
- 固定 896×504 + 多焦距增强（crop→resize）见 [08 §3.4](08_DINO-GeoSemMap_训练计划.md)；深度量程裁 0.2–8m；
- **深度风险最低**：B 缩小 FOV gap，且公开数据多为**真实 metric**，grounding 扎实，Habitat 只需补少量精确 FOV/视角。

### 2.2 检测（阶段1）

| 来源 | 角色 | 提案配比 |
|---|---|---|
| COCO + Objects365 + LVIS | 语义多样性主力，映射到 Top-50 | ~60% |
| **ARKitScenes（自带3D框）/ SUN RGB-D / ScanNet（3D框投2D）** | **in-domain 室内真实视角框**（白捡，缓解视角 gap） | ~25% |
| Habitat（B 内参 + 机位） | in-domain 视角 + 精确 FOV | ~15% |

- **Top-50 映射表**：维护 {源数据集类 → 50 类} 映射（§6，待定稿）；
- ⚠️ **LVIS 联邦/非穷尽标注**：用 federated loss / 忽略未标注区，避免把未标注正样本当负样本压低召回；
- **类别不均衡**：Top-50 家居类长尾，用类别均衡采样；
- 视角 gap 仍在，但 in-domain 室内框 + Habitat 已大幅缓解。

### 2.3 分割（阶段2）

| 来源 | 角色 |
|---|---|
| SAM / SAM2 伪标签（检测框当 prompt） | 主力，规模化、低成本 |
| COCO Panoptic / LVIS instance masks | Top-50 真值 |
| Habitat 实例 mask | in-domain，精确免费 |

- 框内前景/背景二值（[08 §5](08_DINO-GeoSemMap_训练计划.md)）；伪标签做置信度过滤；下游只需 mask 内深度中位数定位物体，精度要求不高。

### 2.4 房间（阶段3）

| 来源                      | 角色        |
| ----------------------- | --------- |
| Matterport3D（region 标注） | 真实房型      |
| Structured3D（合成，房型全）    | 合成房型      |
| HM3D-Semantics          | 真实语义      |
| Habitat 场景房型标签          | in-domain |

- **类目体系（12 类）**：客厅 / 餐厅 / 卧室 / 卫生间 / 厨房 / 书房 / 阳台 / 走廊 / 储物间 / 门厅 / 洗衣房 / 衣帽间（MP3D 映射见 §6.1）；
- ✅ **主卧/次卧已合并为"卧室"**：单帧无法区分主次（外观相同、区别在面积/是否带独立卫，且公开数据只有通用 bedroom）。房间头只输出"卧室"，主/次留给下游 ASM 按面积/布局/是否含独立卫生间推断；
- 过渡区允许低置信度 / 多标签 soft prediction；帧级标签由场景 region 继承。

---

## 3. Habitat 渲染规格

Habitat 的价值不在量大，在 **同域一致 + 精确 FOV + 一次渲染出四 head 全部标签**。从零渲染，参数：

| 项 | 规格 |
|---|---|
| 相机内参 | **B 裁剪后 1280×720 针孔，水平 FOV=90°**：`fx=fy=640, cx=640, cy=360`（垂直 FOV≈58.7°）；Habitat 用 `hfov=90` 等价设定 |
| 机位高度 | **相机离地 1.15 m**（机器人部署高度）；渲染时 agent 相机置于此高度，可加 ±2~3cm 抖动做域随机化 |
| 渲染输出 | RGB + 米制深度 + 实例 mask + 2D/3D 框 + 房间标签（**一次出齐**） |
| 场景 | HM3D / Matterport（高真实度）+ 域随机化（材质/光照/传感器噪声） |
| 轨迹 | 采类机器人运动轨迹/视角 |

> ⚠️ **命门**：Habitat 渲染内参必须 = 真实相机去畸变后在 1280×720 下的针孔内参，sim/real 几何模型才对齐。
> Habitat 是**唯一 in-domain 来源**，在 M0 数据就绪的关键路径上。

---

## 4. 公开数据集 × head 复用表

一份 RGBD 数据集常可喂多个 head，减少搜集量、保证同域：

| 数据集 | 深度 | 检测 | 分割 | 房间 | 真实/合成 |
|---|:--:|:--:|:--:|:--:|---|
| ARKitScenes | ✓✓ | ✓(3D框) | | | 真实 |
| ScanNet / ScanNet++ | ✓✓ | ✓(3D投2D) | ✓ | (场景型) | 真实 |
| Hypersim | ✓✓ | | | | 合成 |
| Matterport3D | ✓ | | | ✓✓ | 真实 |
| SUN RGB-D | ✓ | ✓✓(2D/3D框) | | ✓ | 真实 |
| COCO / O365 / LVIS | | ✓✓ | ✓ | | 真实(网络) |
| Structured3D / HM3D | (✓) | | | ✓✓ | 合成/真实 |
| **Habitat（自渲）** | ✓ | ✓ | ✓ | ✓ | 合成(in-domain) |

---

## 5. in-domain 验证集策略

| 阶段 | eval | 说明 |
|---|---|---|
| 现在 | Habitat held-out（sim→sim） | 每 head 留出渲染集，配合"每轮全 head 确认" |
| 中期 | **小真机 eval（数千~数万帧）** | 你的相机、真实家居；仅做 sim→real 体检，可不参与训练 |

- 深度因有大量真实公开数据，sim→real 风险较低；
- **检测/房间更依赖 in-domain，真机 eval 优先级更高**；
- 短期靠域随机化 + 高真实度 Habitat 场景缩小 gap。

---

## 6. 关键映射表

### 6.1 房间类目（12 类）+ MP3D region 映射

类目：客厅 / 餐厅 / 卧室 / 卫生间 / 厨房 / 书房 / 阳台 / 走廊 / 储物间 / 门厅 / 洗衣房 / 衣帽间

MP3D 30 region 编码 → 12 类（同套映射也用于 Structured3D / HM3D；**提案,可调**）：

| 家居房型 | MP3D 编码 |
|---|---|
| 客厅 | l 客厅 / f 家庭室 / n 休息室 / v 影音 / r 娱乐 |
| 餐厅 | d 餐厅 / D 餐位 |
| 卧室 | b 卧室（主/次合并） |
| 卫生间 | a 浴室 / t 卫生间 |
| 厨房 | k 厨房 |
| 书房 | o 办公室 / i 书房 |
| 阳台 | y 阳台 / p 门廊露台 |
| 走廊 | h 走廊 / s 楼梯 |
| 储物间 | u 工具/杂物间 |
| 门厅 | e 玄关/门厅 |
| 洗衣房 | j 洗衣/杂物 |
| 衣帽间 | c 壁橱/储物 |
| （丢弃） | g 车库 / x 室外 / m 会议 / B 吧台 / w 健身 / C 教室 / S 桑拿 / z 其他 / Z 杂物 / - 无标签 |

> 储物间(u) / 衣帽间(c) 边界较模糊,可按需合并;主卧/次卧处理见 §2.4。

### 6.2 物品类目（起步 30 类，可扩到 Top-50）

> 选类原则：**家居室内 + 导航/房间相关 + 可映射到 COCO/O365/LVIS**。优先大件、定房型的家具/家电（同时是房间 head 的强先验），辅以通行结构（门/窗）与高频软装。按需扩到 Top-50。

| #   | 统一类目            | 中文     | 主要来源（粗→细）                      | 备注                |
| --- | --------------- | ------ | ------------------------------ | ----------------- |
| 1   | sofa            | 沙发     | COCO `couch` / O365 / LVIS     | 客厅强先验             |
| 2   | chair           | 椅子     | COCO `chair` / O365 / LVIS     | 含办公椅、餐椅；LVIS 可再细分 |
| 3   | dining_table    | 餐桌     | COCO `dining table` / O365     | 餐厅强先验             |
| 4   | coffee_table    | 茶几     | O365 `coffee table` / LVIS     | COCO 无，归 O365     |
| 5   | bed             | 床      | COCO `bed` / O365              | 卧室强先验             |
| 6   | nightstand      | 床头柜    | O365 `nightstand` / LVIS       | 卧室                |
| 7   | desk            | 书桌     | O365 `desk` / LVIS             | 书房                |
| 8   | wardrobe        | 衣柜     | O365 `wardrobe` / LVIS         | 卧室/衣帽间            |
| 9   | cabinet         | 橱柜/储物柜 | O365 `cabinet/shelf` / LVIS    | 通用柜体              |
| 10  | bookshelf       | 书架     | O365 `bookshelf` / LVIS        | 书房先验              |
| 11  | tv_stand        | 电视柜    | O365 / LVIS                    | 客厅                |
| 12  | stool           | 凳子     | O365 `stool` / LVIS            |                   |
| 13  | shoe_cabinet    | 鞋柜     | O365 / LVIS                    | 门厅先验              |
| 14  | tv              | 电视     | COCO `tv` / O365               | 客厅/影音强先验          |
| 15  | refrigerator    | 冰箱     | COCO `refrigerator` / O365     | 厨房强先验             |
| 16  | washing_machine | 洗衣机    | O365 `washing machine`         | 洗衣房强先验            |
| 17  | microwave       | 微波炉    | COCO `microwave` / O365        | 厨房                |
| 18  | oven_stove      | 烤箱/灶具  | COCO `oven` / O365 `gas stove` | 厨房                |
| 19  | range_hood      | 抽油烟机   | O365 `extractor` / LVIS        | 厨房                |
| 20  | air_conditioner | 空调     | O365 `air conditioner`         | 墙/窗定位参考           |
| 21  | toilet          | 马桶     | COCO `toilet` / O365           | 卫生间强先验            |
| 22  | sink            | 水槽/洗手盆 | COCO `sink` / O365 `faucet`    | 厨房+卫生间            |
| 23  | bathtub         | 浴缸     | O365 `bathtub` / LVIS          | 卫生间               |
| 24  | door            | 门      | O365 `door` / LVIS             | **导航关键**：房间出入口    |
| 25  | window          | 窗      | O365 / LVIS                    | 阳台/采光定位           |
| 26  | potted_plant    | 盆栽绿植   | COCO `potted plant` / O365     | 高频软装              |
| 27  | lamp            | 灯具     | O365 `lamp` / LVIS             | 含台灯/落地灯           |
| 28  | mirror          | 镜子     | O365 `mirror` / LVIS           | 卫生间/衣帽间           |
| 29  | curtain         | 窗帘     | O365 / LVIS                    | 窗边                |
| 30  | trash_can       | 垃圾桶    | O365 `trash can` / LVIS        | 高频小件              |

映射要点：
- **COCO 直出**（~8 类：sofa/chair/dining_table/bed/tv/refrigerator/microwave/oven/sink/toilet/potted_plant）语义最干净，做主力监督；
- 其余依赖 **O365（细家居）+ LVIS（长尾细分）**，按 §2.2 ⚠️ 用 **federated loss** 处理非穷尽标注，按类别均衡采样压长尾；
- **Habitat in-domain** 渲染时按这 30 类输出 2D/3D 框，补 O365/LVIS 覆盖不到的视角；
- 一类聚合多源标签时维护 `{源类 → 统一类}` 字典（如 LVIS 多种 chair 细类 → `chair`）。

> 扩到 Top-50 的候选（暂留）：clock / vase / pillow / rug(地毯) / picture(挂画) / fan(风扇) / water_heater(热水器) / dishwasher(洗碗机) / kitchen_counter(灶台面) / bidet / towel / curtain_rod / radiator / fireplace / ceiling_light …

---

## 7. 待确认 / TODO

- [x] 房间类目体系：**12 类**（见 §6.1）；主卧/次卧已合并为"卧室"（§2.4）；
- [x] 物品类目：**起步 30 类**（见 §6.2，可扩到 Top-50）；{COCO/O365/LVIS → 30 类} 映射要点已列，细类字典待逐源补全；
- [x] 工作分辨率：**固定 896×504（16:9）**，取消多分辨率；
- [x] B 裁剪后 FOV/内参：**水平 FOV=90°** → 1280×720 下 `fx=fy=640, cx=640, cy=360`（垂直 FOV≈58.7°）；
- [x] f_c 标准焦距：`f_c = 640 × 896/1280 =` **448 px**（896×504 工作分辨率下，主点 cx=448, cy=252）；
- [x] 机器人相机机位高度：**1.15 m**（见 §3）；渲染加 ±2~3cm 抖动域随机化；
- [ ] 各 head 配比的 pilot 实测调整。
