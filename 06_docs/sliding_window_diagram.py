"""
生成滑动窗口流式分块推理策略示意图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib

# 设置中文字体（Windows常用字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# ========== 图1：原始视频帧序列 ==========
ax = axes[0]
ax.set_xlim(0, 100)
ax.set_ylim(0, 2)
ax.set_title('(a) 原始视频帧序列（总帧数 $F$）', fontsize=12, fontweight='bold')

# 绘制帧
frame_colors = ['#3498db'] * 100
for i in range(100):
    rect = mpatches.FancyBboxPatch((i, 0.5), 0.9, 0.8, boxstyle="square,pad=0",
                                    facecolor=frame_colors[i], edgecolor='white', linewidth=0.5)
    ax.add_patch(rect)

ax.text(50, 1.6, r'$F$ 帧', ha='center', fontsize=11)
ax.text(50, 0.2, r'$f_0, f_1, \ldots, f_{F-1}$', ha='center', fontsize=10, color='#2c3e50')
ax.axis('off')

# ========== 图2：滑动窗口切分 ==========
ax = axes[1]
ax.set_xlim(0, 100)
ax.set_ylim(0, 4)
ax.set_title(r'(b) 滑动窗口切分（窗口长度 $L=16$，步长 $S=16$）', fontsize=12, fontweight='bold')

window_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for w in range(5):
    start = w * 16
    color = window_colors[w % len(window_colors)]
    for i in range(16):
        if start + i < 100:
            rect = mpatches.FancyBboxPatch((start + i, 1.5 - w*0.3), 0.9, 0.6,
                                            boxstyle="square,pad=0",
                                            facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.85)
            ax.add_patch(rect)
    ax.annotate(f'$W_{{{w+1}}}$', xy=(start + 8, 1.5 - w*0.3 + 0.3), fontsize=9, ha='center', va='center', color='white', fontweight='bold')

ax.text(50, 0.2, r'$W = \left\lfloor \frac{F - L}{S} \right\rfloor + 1$', ha='center', fontsize=11)
ax.axis('off')

# ========== 图3：流式分块处理 ==========
ax = axes[2]
ax.set_xlim(0, 100)
ax.set_ylim(0, 5)
ax.set_title(r'(c) 流式分块处理（块大小 $C=500$ 窗口 $\rightarrow$ 每块实际读取 $C \times S + L$ 帧）', fontsize=12, fontweight='bold')

chunk_size = 32  # 为了可视化，每块2个窗口（实际500个）
chunk_colors = ['#e74c3c', '#3498db']
for c in range(3):
    start = c * chunk_size
    color = chunk_colors[c % 2]
    for i in range(chunk_size + 16):
        if start + i < 100:
            rect = mpatches.FancyBboxPatch((start + i, 2.5 - c*0.8), 0.9, 0.6,
                                            boxstyle="square,pad=0",
                                            facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
    ax.annotate(f'块 $C_{{{c+1}}}$', xy=(start + chunk_size//2 + 8, 2.5 - c*0.8 + 0.3), fontsize=10, ha='center', va='center', color='white', fontweight='bold')
    # 释放标记
    if c < 2:
        ax.annotate('', xy=(start + chunk_size + 16, 2.5 - c*0.8 + 0.3), xytext=(start + chunk_size + 25, 2.5 - c*0.8 + 0.3),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
        ax.text(start + chunk_size + 28, 2.5 - c*0.8 + 0.3, '释放缓存', fontsize=9, va='center', color='#2c3e50')

ax.text(50, 0.3, r'峰值内存占用恒定：$C \times S + L$ 帧', ha='center', fontsize=11, color='#c0392b', fontweight='bold')
ax.axis('off')

plt.tight_layout(pad=2.0)
plt.savefig('d:\\Projects\\python\\badminton_video_editor\\06_docs\\滑动窗口流式分块推理示意图.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("示意图已保存")
