const PptxGenJS = require("pptxgenjs");

const pres = new PptxGenJS();
pres.layout = "LAYOUT_16x9";
pres.author = "答辩人";
pres.title = "基于动作识别的羽毛球比赛视频回合分割与批量剪辑系统";
pres.subject = "本科毕业设计答辩";

// ===== Color Palette =====
const C = {
  primary: "154360",     // deep navy
  secondary: "1B4F72",   // navy
  accent: "48C9B0",      // teal/mint accent
  darkText: "2C3E50",    // body text
  lightText: "FFFFFF",   // white text
  bgLight: "F4F6F7",     // light gray bg
  bgDark: "154360",      // dark bg
  cardBg: "FFFFFF",      // card white
  muted: "7F8C8D",       // muted text
  border: "D5DBDB",      // light border
  highlight: "E8F6F3",   // light teal highlight
};

const makeShadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.12 });

// ===== Helper: add a section badge =====
function addSectionBadge(slide, text, x, y, w, h, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color } });
  slide.addText(text, { x, y, w, h, color: C.lightText, fontSize: 10, bold: true, align: "center", valign: "middle", fontFace: "Calibri" });
}

// ===== Helper: add a card =====
function addCard(slide, x, y, w, h, fill = C.cardBg) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: C.border, width: 0.5 } });
}

// ===== Helper: accent bar on left of card =====
function addAccentBar(slide, x, y, h, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color } });
}

// ===== Helper: title text =====
function addTitle(slide, text, x, y, w, h, color = C.darkText, size = 32) {
  slide.addText(text, { x, y, w, h, color, fontSize: size, bold: true, fontFace: "Microsoft YaHei", valign: "middle" });
}

// ===== Helper: body bullet text =====
function addBodyText(slide, lines, x, y, w, h, size = 14) {
  const items = lines.map((line, i) => ({ text: line, options: { bullet: true, breakLine: i < lines.length - 1, fontSize: size, color: C.darkText, fontFace: "Microsoft YaHei" } }));
  slide.addText(items, { x, y, w, h, lineSpacing: 22 });
}

// ===================== SLIDE 1: 封面 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgDark };

  // Decorative top accent bar
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: "100%", h: 0.08, fill: { color: C.accent } });

  // Main title
  slide.addText("基于动作识别的羽毛球比赛视频", {
    x: 0.8, y: 1.6, w: 8.4, h: 0.9,
    color: C.lightText, fontSize: 40, bold: true, align: "center", fontFace: "Microsoft YaHei"
  });
  slide.addText("回合分割与批量剪辑系统", {
    x: 0.8, y: 2.4, w: 8.4, h: 0.8,
    color: C.accent, fontSize: 40, bold: true, align: "center", fontFace: "Microsoft YaHei"
  });

  // Subtitle
  slide.addText("—— 设计与实现 ——", {
    x: 0.8, y: 3.3, w: 8.4, h: 0.5,
    color: "BDC3C7", fontSize: 20, align: "center", fontFace: "Microsoft YaHei"
  });

  // Info block at bottom
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 2.5, y: 4.2, w: 5, h: 1.2,
    fill: { color: "1A5276", transparency: 40 },
  });
  slide.addText([
    { text: "答辩人：XXX    学号：XXXXXXXX", options: { breakLine: true, fontSize: 16, color: C.lightText, align: "center", fontFace: "Microsoft YaHei" } },
    { text: "指导老师：XXX    专业：软件工程", options: { fontSize: 16, color: C.lightText, align: "center", fontFace: "Microsoft YaHei" } },
  ], { x: 2.5, y: 4.35, w: 5, h: 1.0 });
}

// ===================== SLIDE 2: 目录 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "目  录", 0.6, 0.4, 2.5, 0.7, C.secondary, 36);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.05, w: 1.2, h: 0.06, fill: { color: C.accent } });

  const toc = [
    { num: "01", title: "研究背景与意义", desc: "行业痛点、技术契机与研究价值" },
    { num: "02", title: "需求分析", desc: "功能需求、非功能需求与角色划分" },
    { num: "03", title: "系统架构与数据库设计", desc: "分层架构、模块划分与核心表结构" },
    { num: "04", title: "关键技术——I3D动作识别", desc: "3D卷积、迁移学习与流式推理" },
    { num: "05", title: "关键技术——YOLOv8检测与协同", desc: "双模型融合、多级优先级与置信度策略" },
    { num: "06", title: "系统演示", desc: "上传、检测、预览、剪辑与管理后台" },
    { num: "07", title: "总结与展望", desc: "成果总结、创新点与未来方向" },
  ];

  toc.forEach((item, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 0.6 + col * 4.6;
    const y = 1.3 + row * 1.05;

    addCard(slide, x, y, 4.3, 0.85, C.cardBg);
    addAccentBar(slide, x, y, 0.85, C.accent);

    slide.addText(item.num, {
      x: x + 0.15, y: y + 0.08, w: 0.7, h: 0.35,
      color: C.accent, fontSize: 20, bold: true, fontFace: "Calibri"
    });
    slide.addText(item.title, {
      x: x + 0.9, y: y + 0.06, w: 3.2, h: 0.35,
      color: C.darkText, fontSize: 16, bold: true, fontFace: "Microsoft YaHei"
    });
    slide.addText(item.desc, {
      x: x + 0.9, y: y + 0.42, w: 3.2, h: 0.35,
      color: C.muted, fontSize: 11, fontFace: "Microsoft YaHei"
    });
  });
}

// ===================== SLIDE 3: 研究背景与意义 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "研究背景与意义", 0.6, 0.35, 4, 0.6, C.secondary, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Left: 3 pain point cards
  const cards = [
    { title: "人工剪辑耗时费力", desc: "一场约50回合的比赛需反复观看，人工标记边界，耗时1小时以上。" },
    { title: "质量难以保证", desc: "长时间观看易疲劳，边界偏移与回合遗漏频发，批量处理更困难。" },
    { title: "现有方案不足", desc: "通用视频分割方法不适用于体育场景，羽毛球专项研究尚处空白。" },
  ];
  cards.forEach((c, i) => {
    const y = 1.25 + i * 1.35;
    addCard(slide, 0.6, y, 4.5, 1.15, C.cardBg);
    addAccentBar(slide, 0.6, y, 1.15, C.secondary);
    slide.addText(c.title, { x: 0.8, y: y + 0.12, w: 4.1, h: 0.35, color: C.darkText, fontSize: 15, bold: true, fontFace: "Microsoft YaHei" });
    slide.addText(c.desc, { x: 0.8, y: y + 0.5, w: 4.1, h: 0.55, color: C.muted, fontSize: 12, fontFace: "Microsoft YaHei" });
  });

  // Right: value block
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 1.25, w: 4.0, h: 4.05, fill: { color: C.secondary } });
  slide.addText("研究价值", { x: 5.7, y: 1.45, w: 3.6, h: 0.5, color: C.accent, fontSize: 20, bold: true, fontFace: "Microsoft YaHei" });

  const values = [
    "理论价值：探索 I3D 时序特征与 YOLO 空间特征的多模态融合策略，为体育视频分析提供新参考。",
    "实践价值：端到端自动化处理，将人工剪辑1小时缩短至30分钟，效率提升约2倍。",
    "应用价值：支持训练复盘、战术分析、赛事制作与教学资源共享，降低使用门槛。"
  ];
  values.forEach((v, i) => {
    slide.addText(v, {
      x: 5.7, y: 2.1 + i * 1.05, w: 3.6, h: 0.9,
      color: C.lightText, fontSize: 13, bullet: true, lineSpacing: 20, fontFace: "Microsoft YaHei"
    });
  });
}

// ===================== SLIDE 4: 需求分析 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "需求分析", 0.6, 0.35, 3, 0.6, C.secondary, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Functional requirements - left
  addCard(slide, 0.6, 1.25, 4.4, 3.9, C.cardBg);
  addAccentBar(slide, 0.6, 1.25, 3.9, C.accent);
  slide.addText("功能性需求", { x: 0.8, y: 1.35, w: 4.0, h: 0.4, color: C.darkText, fontSize: 18, bold: true, fontFace: "Microsoft YaHei" });

  const funcs = [
    "用户认证：注册/登录/密码重置，角色分级（user/admin）",
    "视频管理：上传 MP4/AVI/MOV/MKV（≤2GB），提取元数据",
    "智能检测：异步 I3D+YOLO 推理，轮询进度实时反馈",
    "回合预览：时间线视图 + HTML5 播放器，支持 Range 请求",
    "剪辑下载：FFmpeg 流复制批量剪辑，一键打包导出",
    "管理后台：用户管理、操作日志审计、系统配置"
  ];
  addBodyText(slide, funcs, 0.8, 1.85, 4.0, 3.1, 13);

  // Non-functional + roles - right
  addCard(slide, 5.4, 1.25, 4.2, 1.85, C.cardBg);
  addAccentBar(slide, 5.4, 1.25, 1.85, C.secondary);
  slide.addText("非功能性需求", { x: 5.6, y: 1.35, w: 3.8, h: 0.4, color: C.darkText, fontSize: 18, bold: true, fontFace: "Microsoft YaHei" });
  const nonFuncs = [
    "可用性 ≥ 99.5%，MTBF > 720h",
    "核心操作三次点击内完成",
    "异常自动恢复（GPU 溢出回退 CPU）"
  ];
  addBodyText(slide, nonFuncs, 5.6, 1.85, 3.8, 1.2, 13);

  // Roles
  addCard(slide, 5.4, 3.35, 4.2, 1.8, C.cardBg);
  addAccentBar(slide, 5.4, 3.35, 1.8, C.accent);
  slide.addText("角色划分", { x: 5.6, y: 3.45, w: 3.8, h: 0.4, color: C.darkText, fontSize: 18, bold: true, fontFace: "Microsoft YaHei" });
  slide.addText([
    { text: "普通用户：上传 → 检测 → 预览 → 下载闭环", options: { breakLine: true, fontSize: 13, color: C.darkText, bullet: true, fontFace: "Microsoft YaHei" } },
    { text: "管理员：用户管理、日志审计、系统配置", options: { fontSize: 13, color: C.darkText, bullet: true, fontFace: "Microsoft YaHei" } },
  ], { x: 5.6, y: 3.95, w: 3.8, h: 1.0 });
}

// ===================== SLIDE 5: 系统架构设计 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgDark };

  addTitle(slide, "系统架构设计", 0.6, 0.35, 4, 0.6, C.lightText, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Backend architecture boxes
  const layers = [
    { name: "Web 接入层", y: 1.3, color: "1F618D", items: "Flask + Jinja2 | Session 管理 | RESTful API" },
    { name: "业务逻辑层", y: 2.4, color: "2471A3", items: "用户认证 | AI 推理引擎 | 回合分割 | FFmpeg 剪辑 | 邮件服务" },
    { name: "数据层", y: 3.5, color: "2E86C1", items: "SQLite 数据库 | 本地文件系统 | YAML/JSON 配置" },
  ];
  layers.forEach((layer) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: layer.y, w: 5.0, h: 0.9, fill: { color: layer.color } });
    slide.addText(layer.name, { x: 0.8, y: layer.y, w: 1.6, h: 0.9, color: C.lightText, fontSize: 14, bold: true, valign: "middle", fontFace: "Microsoft YaHei" });
    slide.addText(layer.items, { x: 2.5, y: layer.y + 0.15, w: 2.9, h: 0.6, color: "D6EAF8", fontSize: 11, valign: "middle", fontFace: "Microsoft YaHei" });
  });

  // Arrows between layers
  slide.addShape(pres.shapes.LINE, { x: 3.1, y: 2.2, w: 0, h: 0.2, line: { color: C.accent, width: 2 } });
  slide.addShape(pres.shapes.LINE, { x: 3.1, y: 3.3, w: 0, h: 0.2, line: { color: C.accent, width: 2 } });

  // Frontend architecture - right side
  slide.addShape(pres.shapes.RECTANGLE, { x: 6.0, y: 1.3, w: 3.6, h: 3.1, fill: { color: "1A5276", transparency: 30 } });
  slide.addText("前端分层", { x: 6.2, y: 1.4, w: 3.2, h: 0.4, color: C.accent, fontSize: 16, bold: true, fontFace: "Microsoft YaHei" });

  const frontLayers = [
    "视图层：Jinja2 + Tailwind CSS + Font Awesome",
    "交互控制层：表单校验、进度轮询、播放器控制",
    "数据通信层：Axios + RESTful API 标准化对接"
  ];
  frontLayers.forEach((text, i) => {
    slide.addText(text, { x: 6.2, y: 1.95 + i * 0.75, w: 3.2, h: 0.6, color: C.lightText, fontSize: 12, bullet: true, lineSpacing: 18, fontFace: "Microsoft YaHei" });
  });

  // Deployment note at bottom
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.6, w: 9.0, h: 0.65, fill: { color: "1A5276", transparency: 50 } });
  slide.addText("部署形态：浏览器 → Flask HTTP → PyTorch/YOLOv8/FFmpeg → SQLite/文件系统", {
    x: 0.8, y: 4.65, w: 8.6, h: 0.55, color: "D6EAF8", fontSize: 12, align: "center", valign: "middle", fontFace: "Microsoft YaHei"
  });
}

// ===================== SLIDE 6: 数据库设计 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "数据库设计", 0.6, 0.35, 3, 0.6, C.secondary, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Table
  const tableData = [
    [
      { text: "表名", options: { fill: { color: C.secondary }, color: C.lightText, bold: true, fontSize: 13, align: "center" } },
      { text: "说明", options: { fill: { color: C.secondary }, color: C.lightText, bold: true, fontSize: 13, align: "center" } },
      { text: "核心字段", options: { fill: { color: C.secondary }, color: C.lightText, bold: true, fontSize: 13, align: "center" } },
    ],
    [
      { text: "users", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "用户与权限", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "id, username, password_hash, role, is_active", options: { fontSize: 11, color: C.muted, align: "left" } },
    ],
    [
      { text: "videos", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "视频元信息", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "id, filename, duration, status, created_at", options: { fontSize: 11, color: C.muted, align: "left" } },
    ],
    [
      { text: "operation_logs", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "操作审计", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "id, user_id, action, module, status, created_at", options: { fontSize: 11, color: C.muted, align: "left" } },
    ],
    [
      { text: "email_verification_codes", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "验证码", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "id, email, code, used, expires_at", options: { fontSize: 11, color: C.muted, align: "left" } },
    ],
    [
      { text: "user_statistics", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "用户统计", options: { fontSize: 12, color: C.darkText, align: "center" } },
      { text: "id, user_id, date, videos_uploaded, videos_processed", options: { fontSize: 11, color: C.muted, align: "left" } },
    ],
  ];
  slide.addTable(tableData, {
    x: 0.6, y: 1.25, w: 9.0, h: 2.8,
    border: { pt: 0.5, color: C.border },
    colW: [2.2, 2.0, 4.8],
    fontFace: "Microsoft YaHei",
  });

  // Design highlights below table
  addCard(slide, 0.6, 4.25, 9.0, 1.0, C.cardBg);
  addAccentBar(slide, 0.6, 4.25, 1.0, C.accent);
  slide.addText("设计要点", { x: 0.8, y: 4.32, w: 1.2, h: 0.35, color: C.darkText, fontSize: 14, bold: true, fontFace: "Microsoft YaHei" });
  const dbPoints = [
    "密码采用 SHA-256 加盐哈希存储，禁止明文保存",
    "视频状态机（pending/done）追踪异步检测生命周期",
    "SQLite 单文件零配置部署，满足中小型 Web 应用需求"
  ];
  dbPoints.forEach((p, i) => {
    slide.addText(p, { x: 2.1 + i * 2.6, y: 4.32, w: 2.5, h: 0.85, color: C.muted, fontSize: 11, bullet: true, lineSpacing: 16, fontFace: "Microsoft YaHei" });
  });
}

// ===================== SLIDE 7: I3D动作识别 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "关键技术 —— I3D 动作识别", 0.6, 0.35, 6, 0.6, C.secondary, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Left: key points
  addCard(slide, 0.6, 1.25, 5.0, 3.9, C.cardBg);
  addAccentBar(slide, 0.6, 1.25, 3.9, C.secondary);

  const i3dPoints = [
    { title: "模型原理", desc: "将 2D 卷积核沿时间维度\"膨胀\"为 3D，同时建模空间外观与时间运动信息，主干采用 Inception-v1。" },
    { title: "输入输出", desc: "输入连续 16 帧 RGB 序列，输出二分类标签：round_start（发球）与 round_end（球落地）。" },
    { title: "训练策略", desc: "加载 ImageNet + Kinetics-400 预训练权重做迁移学习，冻结前 5 层，在 78 条样本上验证准确率达 91.67%。" },
    { title: "推理优化", desc: "采用流式分块（streaming chunk）处理长视频，滑动窗口推理，避免内存溢出。" },
  ];
  i3dPoints.forEach((item, i) => {
    const y = 1.4 + i * 0.9;
    slide.addText(item.title, { x: 0.8, y, w: 4.5, h: 0.3, color: C.darkText, fontSize: 14, bold: true, fontFace: "Microsoft YaHei" });
    slide.addText(item.desc, { x: 0.8, y: y + 0.32, w: 4.5, h: 0.45, color: C.muted, fontSize: 11, lineSpacing: 15, fontFace: "Microsoft YaHei" });
  });

  // Right: 3D conv concept diagram
  slide.addShape(pres.shapes.RECTANGLE, { x: 6.0, y: 1.25, w: 3.8, h: 3.9, fill: { color: C.secondary } });
  slide.addText("3D 卷积示意", { x: 6.2, y: 1.4, w: 3.4, h: 0.4, color: C.accent, fontSize: 16, bold: true, align: "center", fontFace: "Microsoft YaHei" });

  // Draw 3D-like boxes to represent 3D convolution
  const boxColors = ["1F618D", "2471A3", "2E86C1", "5DADE2", "85C1E9"];
  boxColors.forEach((col, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 7.0 + i * 0.25, y: 2.2 - i * 0.12, w: 1.6, h: 1.6, fill: { color: col, transparency: 30 } });
  });
  slide.addText("时空特征", { x: 6.8, y: 1.95, w: 2.2, h: 0.35, color: C.lightText, fontSize: 14, bold: true, align: "center", fontFace: "Microsoft YaHei" });

  // Stats callouts
  slide.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 3.6, w: 1.6, h: 0.9, fill: { color: "1A5276", transparency: 40 } });
  slide.addText("91.67%", { x: 6.2, y: 3.65, w: 1.6, h: 0.45, color: C.accent, fontSize: 24, bold: true, align: "center", fontFace: "Calibri" });
  slide.addText("验证准确率", { x: 6.2, y: 4.05, w: 1.6, h: 0.35, color: "D6EAF8", fontSize: 10, align: "center", fontFace: "Microsoft YaHei" });

  slide.addShape(pres.shapes.RECTANGLE, { x: 8.0, y: 3.6, w: 1.6, h: 0.9, fill: { color: "1A5276", transparency: 40 } });
  slide.addText("16", { x: 8.0, y: 3.65, w: 1.6, h: 0.45, color: C.accent, fontSize: 24, bold: true, align: "center", fontFace: "Calibri" });
  slide.addText("输入帧数", { x: 8.0, y: 4.05, w: 1.6, h: 0.35, color: "D6EAF8", fontSize: 10, align: "center", fontFace: "Microsoft YaHei" });
}

// ===================== SLIDE 8: YOLOv8检测与双模型协同 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "关键技术 —— YOLOv8 检测与双模型协同", 0.6, 0.3, 8, 0.6, C.secondary, 30);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.87, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Top: flow diagram
  const boxW = 2.6, boxH = 0.65, startY = 1.15;
  const flowBoxes = [
    { x: 0.6, text: "视频输入", color: "1F618D" },
    { x: 3.5, text: "I3D 动作识别", color: "2471A3" },
    { x: 6.4, text: "YOLOv8 目标检测", color: "1F618D" },
  ];
  flowBoxes.forEach((b) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: b.x, y: startY, w: boxW, h: boxH, fill: { color: b.color } });
    slide.addText(b.text, { x: b.x, y: startY, w: boxW, h: boxH, color: C.lightText, fontSize: 13, bold: true, align: "center", valign: "middle", fontFace: "Microsoft YaHei" });
  });
  // Arrows
  slide.addShape(pres.shapes.LINE, { x: 3.2, y: startY + boxH / 2, w: 0.3, h: 0, line: { color: C.accent, width: 2 } });
  slide.addShape(pres.shapes.LINE, { x: 6.1, y: startY + boxH / 2, w: 0.3, h: 0, line: { color: C.accent, width: 2 } });

  // Middle: decision cards
  const midY = 2.05;
  addCard(slide, 0.6, midY, 4.4, 1.55, C.cardBg);
  addAccentBar(slide, 0.6, midY, 1.55, C.secondary);
  slide.addText("回合开始：三级瀑布策略", { x: 0.8, y: midY + 0.1, w: 4.0, h: 0.35, color: C.darkText, fontSize: 14, bold: true, fontFace: "Microsoft YaHei" });
  slide.addText([
    { text: "L1：运动员左右站位规则", options: { breakLine: true, fontSize: 12, color: C.muted, bullet: true, fontFace: "Microsoft YaHei" } },
    { text: "L2：静止 → 运动状态切换", options: { breakLine: true, fontSize: 12, color: C.muted, bullet: true, fontFace: "Microsoft YaHei" } },
    { text: "L3：I3D round_start 兜底", options: { fontSize: 12, color: C.muted, bullet: true, fontFace: "Microsoft YaHei" } },
  ], { x: 0.8, y: midY + 0.5, w: 4.0, h: 1.0 });

  addCard(slide, 5.4, midY, 4.4, 1.55, C.cardBg);
  addAccentBar(slide, 5.4, midY, 1.55, C.accent);
  slide.addText("回合结束：二级策略", { x: 5.6, y: midY + 0.1, w: 4.0, h: 0.35, color: C.darkText, fontSize: 14, bold: true, fontFace: "Microsoft YaHei" });
  slide.addText([
    { text: "L1：YOLO 检测球落地静止/消失", options: { breakLine: true, fontSize: 12, color: C.muted, bullet: true, fontFace: "Microsoft YaHei" } },
    { text: "L2：I3D round_end 补充判断", options: { breakLine: true, fontSize: 12, color: C.muted, bullet: true, fontFace: "Microsoft YaHei" } },
    { text: "两阶段状态机：等球回手中才触发结束", options: { fontSize: 12, color: C.muted, bullet: true, fontFace: "Microsoft YaHei" } },
  ], { x: 5.6, y: midY + 0.5, w: 4.0, h: 1.0 });

  // Bottom: stats
  const botY = 3.85;
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: botY, w: 9.2, h: 1.45, fill: { color: C.secondary } });

  const stats = [
    { num: "12%~18%", label: "单模型误检率", x: 0.8 },
    { num: "→", label: "", x: 2.6 },
    { num: "7%", label: "双模型融合后", x: 3.4 },
    { num: "91.67%", label: "I3D 验证准确率", x: 5.2 },
    { num: "2×", label: "效率提升", x: 7.0 },
  ];
  stats.forEach((s) => {
    if (s.num === "→") {
      slide.addText(s.num, { x: s.x, y: botY + 0.25, w: 0.6, h: 0.6, color: C.accent, fontSize: 28, bold: true, align: "center", valign: "middle", fontFace: "Calibri" });
    } else {
      slide.addText(s.num, { x: s.x, y: botY + 0.15, w: 1.4, h: 0.55, color: C.accent, fontSize: 26, bold: true, align: "center", valign: "middle", fontFace: "Calibri" });
      slide.addText(s.label, { x: s.x, y: botY + 0.75, w: 1.4, h: 0.35, color: "D6EAF8", fontSize: 11, align: "center", fontFace: "Microsoft YaHei" });
    }
  });
}

// ===================== SLIDE 9: 系统演示 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "系统演示", 0.6, 0.35, 3, 0.6, C.secondary, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  const demos = [
    { title: "视频上传", desc: "支持 MP4/AVI/MOV/MKV\n最大 2GB，实时进度条\n自动提取分辨率/帧率/时长", color: C.secondary },
    { title: "异步检测", desc: "I3D + YOLO 后台推理\n前端每 5% 轮询进度\n异常自动重置状态", color: "2471A3" },
    { title: "回合预览", desc: "时间线视图展示回合\nHTML5 播放器支持拖动\nRange 请求流式播放", color: "1F618D" },
    { title: "剪辑下载", desc: "FFmpeg 流复制批量剪辑\n速度提升 5~10 倍\n一键打包导出", color: C.accent },
  ];

  demos.forEach((d, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.6 + col * 4.7;
    const y = 1.25 + row * 2.15;

    addCard(slide, x, y, 4.5, 1.9, C.cardBg);
    slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.5, h: 0.45, fill: { color: d.color } });
    slide.addText(d.title, { x, y, w: 4.5, h: 0.45, color: C.lightText, fontSize: 15, bold: true, align: "center", valign: "middle", fontFace: "Microsoft YaHei" });
    slide.addText(d.desc, { x: x + 0.2, y: y + 0.6, w: 4.1, h: 1.1, color: C.muted, fontSize: 12, align: "center", valign: "middle", lineSpacing: 20, fontFace: "Microsoft YaHei" });
  });
}

// ===================== SLIDE 10: 总结与展望 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };

  addTitle(slide, "总结与展望", 0.6, 0.35, 3, 0.6, C.secondary, 32);
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.92, w: 1.0, h: 0.05, fill: { color: C.accent } });

  // Left: summary
  addCard(slide, 0.6, 1.25, 4.5, 3.9, C.cardBg);
  addAccentBar(slide, 0.6, 1.25, 3.9, C.secondary);
  slide.addText("工作总结", { x: 0.8, y: 1.35, w: 4.1, h: 0.4, color: C.darkText, fontSize: 18, bold: true, fontFace: "Microsoft YaHei" });

  const summaryItems = [
    "实现基于 I3D + YOLOv8 双模型融合的羽毛球视频智能剪辑系统",
    "验证集准确率达 91.67%，端到端效率较人工提升约 2 倍",
    "提出多级优先级兜底机制与两阶段回合结束状态机",
    "模块化低耦合架构，支持 Web 界面与命令行双模式"
  ];
  summaryItems.forEach((text, i) => {
    slide.addText(text, { x: 0.8, y: 1.9 + i * 0.72, w: 4.1, h: 0.6, color: C.darkText, fontSize: 13, bullet: true, lineSpacing: 18, fontFace: "Microsoft YaHei" });
  });

  // Right: future
  addCard(slide, 5.4, 1.25, 4.4, 3.9, C.cardBg);
  addAccentBar(slide, 5.4, 1.25, 3.9, C.accent);
  slide.addText("未来展望", { x: 5.6, y: 1.35, w: 4.0, h: 0.4, color: C.darkText, fontSize: 18, bold: true, fontFace: "Microsoft YaHei" });

  const futureItems = [
    { title: "扩展动作类别", desc: "引入 SlowFast/MViT，覆盖扣球/吊球/扑球等细粒度动作" },
    { title: "增强实时性", desc: "轻量化模型 + RTMP 直播流接入，实现在线检测" },
    { title: "优化部署体验", desc: "Docker 容器化一键启动，SQLite 迁移至 PostgreSQL" },
    { title: "横向扩展", desc: "向乒乓球、网球等运动推广，构建通用体育视频平台" },
  ];
  futureItems.forEach((item, i) => {
    const y = 1.9 + i * 0.95;
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.7, y, w: 0.3, h: 0.3, fill: { color: C.accent } });
    slide.addText(item.title, { x: 6.15, y, w: 3.4, h: 0.3, color: C.darkText, fontSize: 13, bold: true, fontFace: "Microsoft YaHei" });
    slide.addText(item.desc, { x: 6.15, y: y + 0.32, w: 3.4, h: 0.45, color: C.muted, fontSize: 11, lineSpacing: 15, fontFace: "Microsoft YaHei" });
  });
}

// ===================== SLIDE 11: 致谢 =====================
{
  const slide = pres.addSlide();
  slide.background = { color: C.bgDark };

  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: "100%", h: 0.08, fill: { color: C.accent } });

  slide.addText("致  谢", {
    x: 0.8, y: 1.4, w: 8.4, h: 0.9,
    color: C.lightText, fontSize: 44, bold: true, align: "center", fontFace: "Microsoft YaHei"
  });

  slide.addText("感谢指导老师在选题、架构设计、算法优化与论文撰写过程中的悉心指导；\n感谢学院各位老师的专业课程奠基；\n感谢实验室同学的并肩讨论；\n感谢家人的默默支持。", {
    x: 1.5, y: 2.5, w: 7.0, h: 1.6,
    color: "D6EAF8", fontSize: 16, align: "center", lineSpacing: 26, fontFace: "Microsoft YaHei"
  });

  slide.addText("恳请各位评委老师批评指正！", {
    x: 0.8, y: 4.2, w: 8.4, h: 0.5,
    color: C.accent, fontSize: 20, bold: true, align: "center", fontFace: "Microsoft YaHei"
  });
}

// Save
pres.writeFile({ fileName: "d:/Projects/python/badminton_video_editor/06_docs/答辩PPT.pptx" })
  .then(() => console.log("PPT generated successfully!"))
  .catch((err) => console.error("Error:", err));
