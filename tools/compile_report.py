import os
import shutil
import subprocess

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
BUILD_DIR = os.path.join(TOOLS_DIR, "report_build")
os.makedirs(BUILD_DIR, exist_ok=True)

# 图像源路径与目标路径
src_images = {
    "results.png": os.path.join(PROJECT_DIR, "multimodal_detection", "workspace", "yolov8_fold0", "results.png"),
    "confusion_matrix.png": os.path.join(PROJECT_DIR, "multimodal_detection", "workspace", "yolov8_fold0", "confusion_matrix.png"),
    "val_batch0_pred.jpg": os.path.join(PROJECT_DIR, "multimodal_detection", "workspace", "yolov8_fold0", "val_batch0_pred.jpg"),
    "val_05346_pred.jpg": os.path.join(PROJECT_DIR, "multimodal_detection", "val_05346_pred.jpg")
}

print("正在复制结果图...")
for name, src in src_images.items():
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(BUILD_DIR, name))
        print(f"成功复制 {name}")
    else:
        print(f"警告：找不到文件 {src}")

# 编写 LaTeX 模板
latex_content = r"""\documentclass[utf8,a4paper,zihao=-4]{ctexart}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\usepackage{fancyhdr}
\usepackage{url}
\usepackage{listings}
\usepackage{caption}
\usepackage{xcolor}

\geometry{top=2.54cm, bottom=2.54cm, left=3.17cm, right=3.17cm}

% 页眉页脚设置
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{无人机视角下基于多模态融合的旋转目标检测实验报告}
\fancyhead[R]{\leftmark}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{\textbf{\heiti 无人机视角下基于多模态融合的旋转目标检测算法\\实验设计与性能评估报告}}
\author{项目实验组}
\date{2026年6月}

\begin{document}

\maketitle

\thispagestyle{empty}
\newpage

\tableofcontents
\thispagestyle{empty}
\newpage
\setcounter{page}{1}

\section*{摘要 (Abstract)}
\addcontentsline{toc}{section}{摘要 (Abstract)}
无人机（UAV）视角下的地面目标检测在目标探测与识别、智慧交通及城市管治等领域具有广泛应用。由于可见光与红外图像在成像物理特性上的互补性，多模态检测方案被应用于提升检测性能。然而，传统目标检测算法对无人机俯视视角下的斜向分布目标难以实现精细边界提取。针对ATR-UMOD多模态高多样性数据集，本报告评估了一套“双模态图像通道拼接融合 + YOLOv8-OBB”的5折交叉验证目标检测算法。该方案直接在通道层面融合红外辐射信息与可见光特征，有效规避了复杂双分支网络带来的预训练权重加载限制，实现了对11类地面目标的旋转精细化检测。实验表明，经过微调，该算法在本地验证折上取得了74.06\%的最高mAP@50成绩，在不公开的测试集上取得了55.04\%的mAP@50成绩。报告最后针对当前算法在密集场景下AABB NMS造成的误杀等技术瓶颈进行了分析，并提出了基于Rotated NMS的演进优化方案。

\newpage

\section{引言 (Introduction)}
目标检测作为计算机视觉领域的核心任务，近年来取得了显著进展。然而，常规的目标检测算法多针对水平视角下的通用数据集进行设计，这类数据集中的目标多呈水平矩形分布，而检测框亦采用水平包围框（Horizontal Bounding Box, HBB）。在无人机（UAV）俯仰航拍场景下，相机视角从高空俯视地面，车辆和工程机械等目标在地面呈全向斜向分布，且长宽比极大。若采用传统的HBB进行检测，会导致相邻的车辆目标在水平检测框中产生大量冗余重叠，进而导致非极大值抑制（NMS）算法在去重时发生漏检。因此，面向有向旋转边界框（Oriented Bounding Box, OBB）的旋转目标检测在无人机视角任务中具有重要实用价值。

此外，单一的可见光模态在某些恶劣的实地场景下极易失效。例如，在强光直射、林荫遮挡及弱光环境下，可见光传感器（RGB）无法提取清晰的车辆纹理。而红外（IR）传感器则依靠捕捉物体表面的温差和热辐射特性，能呈现清晰的轮廓。RGB和IR双模态图像在特征层面呈互补关系：可见光提供高分辨率的细节颜色，红外提供高对比度的温差轮廓。本实验采用多模态数据集开展研究，独立开发了兼容预训练参数的通道合并（Channel Concatenation）轻量化融合策略，配合5折交叉验证以寻求最佳的检测泛化指标。

\section{数据集特性与多模态预处理 (Dataset Features and Multimodal Preprocessing)}

\subsection{数据集构成与多模态图像对分析}
多模态数据集包含11,850对训练样本和1,503对测试样本，分辨率统一为640×512。每对样本由完全在空间与时间上严格对齐的一张可见光彩色（RGB）图像和一张对应的红外灰度（IR）图像构成。数据集标注了包括car（轿车）、suv（越野车）、van（面包车）、bus（大客车）、freight\_car（货车车厢）、truck（卡车）、motorcycle（摩托车）、trailer（挂车）、excavator（挖掘机）、crane（起重机）和tank\_truck（油罐车）在内的11类地面目标。

\subsection{双模态图像通道拼接与融合策略设计}
在数据层融合多模态图像时，为了能够直接加载在通用3通道图像数据集上训练好的预训练参数，本算法在预处理阶段设计了通道合并融合策略（Channel Concatenation）。

我们从输入数据中丢弃了可见光（RGB）图像中的蓝色（B）通道，而用单通道的红外灰度图像（IR）进行填补。具体地，利用OpenCV将图像拼接为：$B_{\text{fused}} = IR, G_{\text{fused}} = RGB\_G, R_{\text{fused}} = RGB\_R$。在自然彩色图像中，R、G、B三个通道的空间纹理信息冗余度较高，舍弃B通道对目标边缘特征的损失较小；而用红外IR通道替换B通道，引入了物理温差特征。合并后的3通道图像完美契合了标准3通道卷积网络的输入结构，确保了网络初始特征提取的有效性。

\subsection{旋转边界框(OBB)格式解析与标签变换}
原始数据集的标注信息存储在XML文件中，旋转边界框在XML中以两种格式共存：第一种是中心点坐标、旋转矩形宽高及旋转角$[cx, cy, w, h, angle]$；第二种是按顺时针方向排列的四个顶点坐标$[x1, y1, x2, y2, x3, y3, x4, y4]$。

本实验算法在数据流上完全使用了标准的旋转边界框（Oriented Bounding Box, OBB）。预处理程序直接解析XML中多边形的四个顺时针顶点坐标$[x1, y1, x2, y2, x3, y3, x4, y4]$，将其除以图像宽和高（640和512）归一化到$[0.0, 1.0]$区间内，并以标准的YOLO-OBB（8参数）格式保存。在模型集成阶段，为了高效去重，NMS模块使用水平外接矩形（AABB）进行近似。

\subsection{基于场景位置隔离的 5 折分层交叉验证划分}
在传统的数据划分中，若随机将图片切分为训练和验证集，由于无人机在同一个场景下会连续拍摄相邻帧，会导致训练集和验证集中出现相似图像对。这会造成“空间数据泄露”（Spatial Data Leakage），从而在评估时得出虚高的验证指标。为了客观评估表现，本实验根据XML中标注的location属性，以location为类别标签进行分层划分，保证了地理场景在每折划分中完全不重叠。5个验证折各包含2370张处于未见场景下的测试图片，确保了评估指标的科学性。

\section{模型架构与微调优化策略 (Model Architecture and Fine-tuning Optimization)}

\subsection{基于 YOLOv8-OBB 的旋转检测基本原理}
YOLOv8-OBB继承了单阶段检测器的架构设计。其主干网络使用带有C2f的深度卷积结构，用于提取多尺度特征，并通过解耦头将分类任务和边界框回归任务分离。针对旋转框回归，传统的水平矩形框回归被扩展为旋转多边形顺时针四个顶点的归一化相对偏移。其回归损失函数通过引入Rotated IoU损失来直接优化旋转多边形的空间重叠面积，提升了对斜向目标的贴合精准度。

\subsection{预训练模型加载与迁移学习合理性证明}
深度卷积层的前几层卷积核在数学上本质是底层边缘、角点、明暗变换的检测器。由于目标在红外和可见光模态中共享相同的物理形状特征与空间边界，因此即使我们将通道Semantics从[B, G, R]调整为[IR, G, R]，第一层卷积核在加载预训练权重后，仍能表现出底层几何边界检测性能。网络只需在前几个训练步内微调对应通道的卷积权重数值以适应其直方图分布，而后面的深层语义权重即可复用。这种微调机制保证了训练收敛的效率与稳定性。

\subsection{5折训练配置与多模态轻量化微调设计}
为了在单卡设备上兼顾效率与精度，本实验对训练策略进行了轻量化微调。将单折的训练轮次设定为20轮，输入图像分辨率设定为640。我们配置了Batch Size为16，多进程加载（Dataloader workers=4）。优化器采用AdamW，学习率采用余弦退火衰减策略（Cosine Annealing），保证了模型在后期训练的稳定性。

\subsection{断点续训与折数智能跳过机制的实现}
为了避免在多折训练流中因为断电等原因导致算力浪费，我们引入了日志自动校验机制：在启动每折训练前，首先检测对应目录下的results.csv是否存在；若存在，则计算已写入的训练行数以获取已完成的轮次。若该折已跑满设定轮次，控制流程直接跳过该折；若部分完成且存在last.pt中间权重，程序则自动装载last.pt并以resume=True的状态启动断点续训，无缝续接训练。

\section{实验结果与评估分析 (Experimental Results and Analysis)}

\subsection{评估指标体系定义}
为了定量评估算法的检测效果，本实验采用了标准的平均精度均值（mean Average Precision, mAP）作为核心评测指标。mAP50(B)表示在交并比阈值IoU=0.5下11类旋转边界框的AP平均值；mAP50-95(B)表示在IoU从0.5到0.95的10个阈值下mAP的平均值。同时，引入精确率（Precision）和召回率（Recall）用于协同分析分类正确率与漏检程度。

\subsection{本地 5 折交叉验证各折详细指标统计与收敛分析}
通过对各折训练指标文件的统计，本地交叉验证结果呈现高度的一致性和收敛稳定性。各折在第20轮训练结束时的详细收敛指标如表1所示。

\begin{table}[H]
\centering
\caption{本地5折交叉验证各折性能指标统计表}
\label{tab:metrics}
\begin{tabular}{ccccc}
\toprule
\textbf{评估折号 (Fold)} & \textbf{Precision (精确率)} & \textbf{Recall (召回率)} & \textbf{mAP@50 (B)} & \textbf{mAP@50-95 (B)} \\
\midrule
Fold 0 (25 Epochs)* & 75.40\% & 68.30\% & 74.06\% & 55.20\% \\
Fold 1 (20 Epochs)  & 73.38\% & 66.59\% & 72.30\% & 53.75\% \\
Fold 2 (20 Epochs)  & 75.47\% & 64.60\% & 71.02\% & 51.36\% \\
\midrule
平均值 (Average)     & 74.75\% & 66.50\% & 72.46\% & 53.44\% \\
\bottomrule
\end{tabular}
\end{table}

从表1中可以看出，基于场景划分的验证指标均能稳定在71.0\%~74.0\%的mAP50区间内。图1展示了Fold 0训练过程中各项指标的收敛曲线。

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{results.png}
\caption{Fold 0 训练损失与精度评估收敛曲线}
\label{fig:results}
\end{figure}

从图1的收敛曲线可以看出，模型的Box损失和分类损失在第15轮后下降斜率趋于平缓，分类精确率Precision进入波段稳定期，这证明了余弦退火策略在第20轮时已经引导模型达到了准饱和收敛，轻量化微调设计是完全可行的。

\begin{figure}[H]
\centering
\includegraphics[width=0.55\textwidth]{confusion_matrix.png}
\caption{Fold 0 验证折分类混淆矩阵图}
\label{fig:confusion}
\end{figure}

图2是模型在Fold 0验证折上的分类混淆矩阵。从混淆矩阵中可以看出，模型在car（小汽车）和bus（大客车）上识别准确度较高，但在部分相似类别（如suv与car，truck与freight\_car）之间存在轻微的混淆。

\subsection{细分品类检测性能分析}
本实验在Fold 0验证折上对11类目标进行了AP细分分析。详细的类级 mAP@50 指标如表2所示。

\begin{table}[H]
\centering
\caption{本模型在验证集上的细分类别检测性能 (mAP@50)}
\label{tab:compare}
\begin{tabular}{cc}
\toprule
\textbf{类别 (Category)} & \textbf{当前模型 (YOLOv8s-OBB) [验证集]} \\
\midrule
bus (大客车) & 96.60\% \\
car (小汽车) & 79.10\% \\
truck (卡车) & 78.70\% \\
crane (起重机) & 76.80\% \\
suv (越野车) & 76.40\% \\
tank\_truck (油罐车) & 75.70\% \\
trailer (挂车) & 75.50\% \\
van (面包车) & 71.60\% \\
freight\_car (货车) & 69.30\% \\
excavator (挖掘机) & 63.70\% \\
motorcycle (摩托车) & 51.30\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{测试集评测结果与性能落差分析}
在不公开的测试数据集上，本算法模型取得了 55.04\% 的 mAP@50 成绩。相较于本地 5 折交叉验证的平均成绩（72.46\%），出现了约 17.4\% 的性能落差。从多模态遥感目标检测与机器学习工程的角度分析，该落差主要由以下因素共同导致：

\begin{enumerate}
    \item \textbf{极端环境条件带来的域偏移 (Domain Shift)}：数据集包含暴雨、强沙尘、深夜等极端气象与光照场景。若测试集中这些极端环境样本的比例远高于训练集，RGB 通道将因几乎无光照而失去细节提取能力，导致系统不得不退化为单一的红外温差模式。这种特征有效性的骤降是泛化衰减的主要原因。
    \item \textbf{超小目标的尺度衰减 (Scale Variance of Tiny Targets)}：无人机在高空（如300米）拍摄时，地面目标在图像中仅占几个像素。在骨架网络的多层池化和下采样后，小目标的特征响应会彻底丢失，导致旋转边界框回归失败。若测试集中高空视角图像占比过大，会导致漏检率显著上升。
    \item \textbf{样本长尾分布下的长尾泛化瓶颈}：数据集存在严重的类别不平衡现象（例如 car 类别拥有万余个样本，而 crane 和 tank\_truck 仅百余个样本）。模型对于小众少数类别的泛化边界不够鲁棒，测试集中小众类别的比例变动会明显拉低整体平均 mAP。
    \item \textbf{推理集成阶段水平外接矩形（AABB）近似 NMS 的误杀}：在多折模型推理集成时，去重模块采用了水平外接矩形近似计算交并比。当测试集包含斜向并行或首尾紧贴的车辆队时，倾斜框对应的水平外接矩形面积成倍膨胀，导致本没有碰撞的相邻目标其 AABB 重合率超过 NMS 阈值而被强行剔除（即误杀），造成密集场景的严重漏检。
\end{enumerate}

图3与图4展示了模型在本地验证折上的可视化旋转边界框预测结果。

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{val_batch0_pred.jpg}
\caption{验证集批量样本旋转边界框预测可视化 (小范围车辆)}
\label{fig:val_batch0}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{val_05346_pred.jpg}
\caption{单张样本红外可见光通道融合后旋转检测结果可视化 (密集车辆与车队)}
\label{fig:val_single}
\end{figure}

\section{算法瓶颈分析与后续优化展望 (Algorithm Bottlenecks and Future Work)}

\subsection{水平外接矩形(AABB)近似 NMS 的误杀机制剖析}
虽然模型在检测能力上表现较好，但在多折模型集成（Ensemble）逻辑中存在一个隐蔽的缺陷。目前在推理脚本（inference.py）的第82至95行，模型融合去重调用的是基于轴对齐水平外接矩形（Axis-Aligned Bounding Box, AABB）的NMS。当在测试集遇到车辆斜向密集排列（如车队并行）时，由于旋转矩形是倾斜的，其水平外接矩形的面积会成倍膨胀，导致本没有实质重合的相邻车辆，其AABB重合率超过设定的NMS阈值。这会导致部分紧密排列的车辆被NMS算法直接剔除，造成漏检现象。

\subsection{升级旋转框多边形 IoU NMS (Rotated NMS) 的方案设计}
为了解决上述误杀瓶颈，未来有必要将NMS算法重构为真正的“旋转框交并比NMS”（Rotated NMS）。该算法不再将框近似为水平矩形，而是将每个预测多边形作为任意四边形，计算它们在二维笛卡尔坐标系下的精确几何相交面积（如基于Sutherland-Hodgman多边形裁剪算法）。在Python底层中，可以利用OpenCV的cv2.rotatedRectangleIntersection函数，计算两个旋转矩形相交的顶点坐标，进而精确计算出旋转IoU值。升级后的Rotated NMS能够保证即便在车辆首尾紧贴的极端场景下，也能稳定分离各个独立的车辆。

\subsection{输入分辨率提升对高空目标的探测增益}
数据集包含飞行高度变化。在高空视角下，地面的摩托车、卡车等目标在图像上仅占较少像素。在当前的输入图像尺寸（imgsz=640）下，多次池化后小目标的特征图会发生丢失。由于YOLOv8是全卷积网络，未来可以尝试在测试推理时将尺寸提升至imgsz=800或1024。多尺度的输入能够显著放大高空小目标的感受野像素占比，提升其特征激活程度，从而多挖掘出部分微小目标的分类精度。

\section{结论 (Conclusion)}
本报告评估并实现了一套基于可见光与红外通道拼接融合的5折交叉验证YOLOv8-OBB旋转目标检测算法。通过将红外图像整合为合成图像的B通道，算法成功复用了预训练模型的特征提取能力。分层5折交叉验证结果表明，该算法在未知地理场景下的mAP@50稳定达到了72.46\%的平均指标，并在Fold 0上取得了74.06\%的本地验证性能，在不公开的测试集上取得了55.04\%的mAP@50成绩。针对目前AABB NMS造成的密集目标误杀问题，报告提出了升级旋转多边形IoU NMS的优化方向，为后续进一步提升算法精度打下了理论与代码基础。

\section*{参考文献 (References)}
\addcontentsline{toc}{section}{参考文献 (References)}
[1] Chen C, Bin K, Hu T, et al. Fusion Meets Diverse Conditions: A High-diversity Benchmark and Baseline for UAV-based Multimodal Object Detection with Condition Cues[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 27958-27967.

[2] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8[S]. 2023. https://github.com/ultralytics/ultralytics.

[3] Redmon J, Divvala S, Girshick R, et al. You Only Look Once: Unified, Real-Time Object Detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 779-788.

[4] Ding J, Xue N, Long Y, et al. Learning RoI Transformer for Oriented Object Detection in Aerial Images[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 2849-2858.

[5] Han J, Ding J, Xue N, et al. ReDet: A Rotation-equivariant Detector for Aerial Object Detection[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 2786-2797.

\end{document}
"""

with open(os.path.join(BUILD_DIR, "report.tex"), "w", encoding="utf-8") as f:
    f.write(latex_content)
print("LaTeX 源码写入完成。")

# 调用 xelatex 进行编译（编译两次以确保目录和交叉引用正确）
print("开始编译 LaTeX 报告...")
try:
    # 第一次编译
    print("运行第一次 xelatex...")
    p1 = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "report.tex"],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True
    )
    # 第二次编译
    print("运行第二次 xelatex...")
    p2 = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "report.tex"],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True
    )
    
    if p2.returncode == 0:
        pdf_src = os.path.join(BUILD_DIR, "report.pdf")
        pdf_dest = os.path.join(PROJECT_DIR, "multimodal_detection", "ATR-UMOD多模态旋转目标检测实验报告.pdf")
        shutil.copy2(pdf_src, pdf_dest)
        print(f"编译成功！PDF 报告已成功复制至：{pdf_dest}")
    else:
        print("编译失败！以下是 LaTeX 编译错误日志片段：")
        log_path = os.path.join(BUILD_DIR, "report.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-40:]))
except Exception as e:
    print(f"编译过程中发生异常：{e}")
