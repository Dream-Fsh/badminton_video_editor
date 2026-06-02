"""
羽毛球视频自动剪辑系统 - Web前端
现代化的Web界面，支持视频上传、ROI选择、智能检测和回合播放
"""

import os
import sys
import json
import uuid
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session, redirect
from werkzeug.utils import secure_filename
from functools import wraps
import cv2
import numpy as np
from datetime import datetime

# ==================== 并发检测任务管理 ====================
# 支持多个用户同时执行检测任务，每个任务通过 task_id 隔离
MAX_CONCURRENT_TASKS = 5  # 最大并发检测任务数

# 全局任务存储: { task_id: { 'status', 'progress', 'message', 'result', 'error', 'thread', 'created_at' } }
detection_tasks = {}
detection_tasks_lock = threading.Lock()  # 保护 detection_tasks 字典的读写


def _create_task(video_info, model_path):
    """创建并启动一个独立的检测任务，返回 task_id"""
    task_id = str(uuid.uuid4())

    task = {
        'status': 'running',
        'progress': 0,
        'message': '正在初始化...',
        'result': None,
        'error': None,
        'thread': None,
        'created_at': datetime.now()
    }

    # 定义该任务专属的进度回调
    def progress_callback(percent, message):
        with detection_tasks_lock:
            if task_id in detection_tasks:
                detection_tasks[task_id]['progress'] = percent
                detection_tasks[task_id]['message'] = message

    # 该任务的后台线程
    def run_detection():
        try:
            from model_predict_optimized import ActionPredictorFast as ActionPredictor
            predictor = ActionPredictor(model_path)

            full_results = predictor.predict_video(
                video_info['filepath'], sliding_stride=8, progress_callback=progress_callback
            )
            rounds = full_results.get('rounds', [])

            with detection_tasks_lock:
                if task_id not in detection_tasks:
                    return  # 任务已被清理

                all_raw_rounds = full_results.get('all_raw_rounds', [])
                auto_approved_ids = full_results.get('auto_approved_ids', [])

                if not rounds:
                    detection_results = {
                        'video_file': video_info['filename'],
                        'total_duration': video_info['duration'],
                        'detected_rounds': [],
                        'all_raw_rounds': all_raw_rounds,
                        'auto_approved_ids': auto_approved_ids,
                        'detection_time': datetime.now().isoformat(),
                        'statistics': {'total_rounds': 0}
                    }
                else:
                    detection_results = {
                        'video_file': video_info['filename'],
                        'total_duration': video_info['duration'],
                        'detected_rounds': rounds,
                        'all_raw_rounds': all_raw_rounds,
                        'auto_approved_ids': auto_approved_ids,
                        'detection_time': datetime.now().isoformat(),
                        'statistics': {
                            'total_rounds': len(rounds),
                            'total_active_time': sum(r['duration'] for r in rounds),
                            'average_round_duration': sum(r['duration'] for r in rounds) / len(rounds),
                            'detection_accuracy': 1.0
                        }
                    }

                detection_tasks[task_id]['status'] = 'completed'
                detection_tasks[task_id]['result'] = detection_results
                detection_tasks[task_id]['progress'] = 100
                detection_tasks[task_id]['message'] = '识别完成'
                
                # 记录视频处理统计
                from datetime import date
                user_id = video_info.get('user_id')
                if user_id:
                    increment_user_stat(user_id, date.today(), 'videos_processed')
                
        except Exception as e:
            with detection_tasks_lock:
                if task_id in detection_tasks:
                    detection_tasks[task_id]['status'] = 'error'
                    detection_tasks[task_id]['error'] = str(e)
            import traceback
            traceback.print_exc()

    # 注册任务
    with detection_tasks_lock:
        detection_tasks[task_id] = task

    # 启动后台线程
    t = threading.Thread(target=run_detection)
    t.daemon = True
    t.start()

    with detection_tasks_lock:
        detection_tasks[task_id]['thread'] = t

    return task_id


def _cleanup_old_tasks():
    """清理已完成或出错的旧任务（保留最近 50 条记录）"""
    with detection_tasks_lock:
        # 先收集所有非运行中的 task_id
        finished = [
            (tid, t['created_at'])
            for tid, t in detection_tasks.items()
            if t['status'] in ('completed', 'error', 'idle')
        ]
        # 按创建时间排序，保留最新的 50 条
        finished.sort(key=lambda x: x[1], reverse=True)
        to_remove = [tid for tid, _ in finished[50:]]
        for tid in to_remove:
            detection_tasks.pop(tid, None)

# 导入认证模块
import sys
sys.path.insert(0, str(Path(__file__).parent))
from auth import authenticate_user, register_user, log_operation, get_all_users, get_user_by_id, update_user_status, delete_user, get_operation_logs, get_user_count, reset_user_password, find_user_by_username, generate_verification_code, store_verification_code, verify_code, get_user_email, get_db, increment_user_stat
from email_service import send_password_reset_email, email_service, test_email_service

# 获取项目根目录（web_app.py 在 02_code/ 下，需要回到上级目录）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)  # 切换工作目录到项目根目录

app = Flask(__name__, 
            template_folder=str(PROJECT_ROOT / 'templates'),
            static_folder=str(PROJECT_ROOT / 'static'))
app.secret_key = 'badminton_video_editor_2024'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size

# 配置目录（使用绝对路径避免工作目录变化导致的问题）
UPLOAD_FOLDER = str(PROJECT_ROOT / '01_data' / 'raw_videos')
PROCESSED_FOLDER = str(PROJECT_ROOT / '04_output')
TEMP_FOLDER = str(PROJECT_ROOT / 'temp')

# 确保目录存在
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ==================== 系统设置管理 ====================

SETTINGS_FILE = PROJECT_ROOT / 'config' / 'system_settings.json'

def load_system_settings():
    """加载系统设置"""
    default_settings = {
        'site_name': '羽毛球视频智能剪辑系统',
        'allow_register': True,
        'max_upload_size_mb': 2048,
        'allowed_video_formats': ['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        'email': {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'sender_email': '',
            'sender_password': '',
            'sender_name': '羽毛球视频剪辑系统'
        }
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            # 递归合并
            def merge_dict(base, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        merge_dict(base[key], value)
                    else:
                        base[key] = value
                return base
            return merge_dict(default_settings, saved)
        except Exception:
            pass
    return default_settings

def save_system_settings(settings):
    """保存系统设置"""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

# 加载一次设置到内存
_system_settings = load_system_settings()

def get_current_settings():
    """获取当前系统设置（支持热加载）"""
    global _system_settings
    _system_settings = load_system_settings()
    return _system_settings

def get_allowed_extensions():
    """获取允许的视频格式集合"""
    settings = get_current_settings()
    formats = settings.get('allowed_video_formats', ['mp4', 'avi', 'mov', 'mkv', 'wmv'])
    return set(f.lower() for f in formats)

def get_max_upload_size():
    """获取最大上传大小（字节）"""
    settings = get_current_settings()
    return settings.get('max_upload_size_mb', 2048) * 1024 * 1024

# 允许的视频格式
ALLOWED_EXTENSIONS = get_allowed_extensions()

def allowed_file(filename):
    extensions = get_allowed_extensions()
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# ==================== 登录验证装饰器 ====================

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': '请先登录'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': '请先登录'}), 401
            return redirect('/login')
        if session.get('user_role') != 'admin':
            if request.is_json:
                return jsonify({'error': '需要管理员权限'}), 403
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function

# ==================== 认证相关路由 ====================

@app.route('/login')
def login_page():
    """登录页面"""
    if 'user_id' in session:
        if session.get('user_role') == 'admin':
            return redirect('/admin')
        return redirect('/')
    return render_template('login.html')

@app.route('/register')
def register_page():
    """注册页面"""
    if 'user_id' in session:
        return redirect('/')
    return render_template('register.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """用户登录API"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'}), 400
        
        result = authenticate_user(username, password)
        
        if result['success']:
            # 保存用户信息到session
            session['user_id'] = result['user']['id']
            session['username'] = result['user']['username']
            session['user_role'] = result['user']['role']
            
            # 记录登录统计
            from datetime import date
            increment_user_stat(result['user']['id'], date.today(), 'login_count')
            
            return jsonify({
                'success': True,
                'user': result['user']
            })
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    """用户注册API"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        email = data.get('email', '').strip()
        
        # 基本验证
        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'}), 400
        
        if not email:
            return jsonify({'success': False, 'error': '邮箱不能为空'}), 400
        
        if len(username) < 3:
            return jsonify({'success': False, 'error': '用户名至少需要3个字符'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': '密码至少需要6个字符'}), 400
        
        # 检查是否允许注册
        settings = get_current_settings()
        if not settings.get('allow_register', True):
            return jsonify({'success': False, 'error': '当前系统已关闭用户注册'}), 403
        
        result = register_user(username, password, email)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': '注册成功！请登录'
            })
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """用户登出API"""
    username = session.get('username')
    user_id = session.get('user_id')
    
    # 记录日志
    log_operation(user_id, 'logout', f'用户登出: {username}')
    
    # 清除session
    session.clear()
    
    return jsonify({'success': True, 'message': '已退出登录'})

@app.route('/api/current_user')
def api_current_user():
    """获取当前登录用户信息"""
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user': {
                'id': session.get('user_id'),
                'username': session.get('username'),
                'role': session.get('user_role')
            }
        })
    return jsonify({'logged_in': False})

# ==================== 忘记密码路由 ====================

@app.route('/forgot-password')
def forgot_password_page():
    """忘记密码页面"""
    if 'user_id' in session:
        return redirect('/')
    return render_template('forgot_password.html')

@app.route('/api/forgot-password/send-code', methods=['POST'])
def api_send_verification_code():
    """发送验证码"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        
        if not username:
            return jsonify({'success': False, 'error': '请输入用户名'}), 400
        
        # 查找用户
        user = find_user_by_username(username)
        if not user:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
        
        # 生成验证码
        code = generate_verification_code(6)
        store_verification_code(username, code)
        
        # 获取用户邮箱
        user_email = get_user_email(username)
        
        # 使用 SMTP 邮件服务
        if email_service.is_enabled and user_email:
            # SMTP 模式
            success = send_password_reset_email(user_email, username, code)
            if success:
                return jsonify({
                    'success': True,
                    'message': '验证码已发送到您的注册邮箱',
                    'dev_mode': False,
                    'email_service': 'smtp',
                    'username': username
                })
            else:
                return jsonify({'success': False, 'error': '邮件发送失败，请稍后重试'}), 500
        
        # 开发模式：直接返回验证码
        else:
            return jsonify({
                'success': True,
                'message': '验证码已生成（开发模式）',
                'dev_mode': True,
                'code': code,
                'username': username
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/forgot-password/reset', methods=['POST'])
def api_reset_password():
    """重置密码"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        code = data.get('code', '').strip()
        new_password = data.get('new_password', '')
        
        if not username or not code or not new_password:
            return jsonify({'success': False, 'error': '参数不完整'}), 400
        
        if len(new_password) < 6:
            return jsonify({'success': False, 'error': '密码至少6个字符'}), 400
        
        # 验证验证码
        if not verify_code(username, code):
            return jsonify({'success': False, 'error': '验证码错误或已过期'}), 400
        
        # 获取用户并重置密码
        user = find_user_by_username(username)
        if not user:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
        
        reset_user_password(user['id'], new_password)
        
        return jsonify({
            'success': True,
            'message': '密码重置成功，请使用新密码登录'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== 管理员路由 ====================

@app.route('/admin')
@admin_required
def admin_page():
    """管理员后台页面"""
    return render_template('admin.html')

@app.route('/admin/users')
@admin_required
def admin_users():
    """用户管理页面"""
    users = get_all_users()
    return jsonify({'success': True, 'users': users})

@app.route('/admin/logs')
@admin_required
def admin_logs():
    """操作日志"""
    logs = get_operation_logs(100)
    return jsonify({'success': True, 'logs': logs})

@app.route('/admin/api/stats')
@admin_required
def admin_stats():
    """管理员统计信息"""
    user_count = get_user_count()
    
    # 获取统计数据
    conn = get_db()
    cursor = conn.cursor()
    
    # 视频数量
    cursor.execute("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'")
    
    conn.close()
    
    return jsonify({
        'success': True,
        'stats': {
            'user_count': user_count,
            'video_count': len([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]),
            'output_count': len([f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.mp4')]),
            'email_enabled': email_service.is_enabled
        }
    })

@app.route('/admin/api/user-stats')
@admin_required
def admin_user_stats():
    """用户活跃度统计 - 最近7天数据"""
    from datetime import date, timedelta
    conn = get_db()
    cursor = conn.cursor()
    
    # 最近7天的日期列表
    dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    
    # 按日汇总统计数据
    cursor.execute('''
        SELECT date, 
               SUM(videos_uploaded) as videos_uploaded,
               SUM(videos_processed) as videos_processed,
               SUM(login_count) as login_count
        FROM user_statistics
        WHERE date >= ?
        GROUP BY date
        ORDER BY date ASC
    ''', (dates[0],))
    rows = cursor.fetchall()
    conn.close()
    
    # 构建日期到数据的映射
    data_map = {row['date']: dict(row) for row in rows}
    
    daily_stats = []
    for d in dates:
        daily_stats.append({
            'date': d,
            'videos_uploaded': data_map.get(d, {}).get('videos_uploaded', 0) or 0,
            'videos_processed': data_map.get(d, {}).get('videos_processed', 0) or 0,
            'login_count': data_map.get(d, {}).get('login_count', 0) or 0
        })
    
    return jsonify({
        'success': True,
        'dates': dates,
        'daily': daily_stats
    })

@app.route('/admin/api/email/test', methods=['POST'])
@admin_required
def admin_test_email():
    """测试邮件服务连接"""
    # 测试 SMTP
    if email_service.is_enabled:
        result = test_email_service()
        result['service'] = 'smtp'
        return jsonify(result)
    else:
        return jsonify({'success': False, 'message': '未启用邮件服务'})

@app.route('/admin/api/user/<int:user_id>/reset-password', methods=['POST'])
@admin_required
def admin_reset_user_password(user_id):
    """管理员重置用户密码"""
    data = request.json
    new_password = data.get('new_password', '')
    
    if not new_password or len(new_password) < 6:
        return jsonify({'success': False, 'error': '密码至少需要6个字符'}), 400
    
    reset_user_password(user_id, new_password)
    log_operation(session.get('user_id'), 'reset_password', f'管理员重置用户 {user_id} 的密码')
    
    return jsonify({
        'success': True,
        'message': '密码重置成功'
    })

@app.route('/admin/api/user/<int:user_id>/status', methods=['PUT'])
@admin_required
def admin_update_user_status(user_id):
    """更新用户状态"""
    data = request.json
    is_active = data.get('is_active', 1)
    
    update_user_status(user_id, is_active)
    log_operation(session.get('user_id'), 'update_user', f'更新用户状态: {user_id} -> {is_active}')
    
    return jsonify({'success': True})

@app.route('/admin/api/user/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(user_id):
    """删除用户"""
    # 防止删除自己
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'error': '不能删除自己'}), 400
    
    delete_user(user_id)
    log_operation(session.get('user_id'), 'delete_user', f'删除用户: {user_id}')
    
    return jsonify({'success': True})

@app.route('/admin/api/settings', methods=['GET'])
@admin_required
def admin_get_settings():
    """获取系统设置"""
    settings = get_current_settings()
    # 隐藏邮箱密码
    safe_settings = json.loads(json.dumps(settings))
    if 'email' in safe_settings and 'sender_password' in safe_settings['email']:
        safe_settings['email']['sender_password'] = '********' if safe_settings['email']['sender_password'] else ''
    return jsonify({'success': True, 'settings': safe_settings})

@app.route('/admin/api/settings', methods=['POST'])
@admin_required
def admin_save_settings():
    """保存系统设置"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': '请求数据为空'}), 400
        
        current = get_current_settings()
        
        # 更新基本设置
        if 'site_name' in data:
            current['site_name'] = str(data['site_name']).strip() or '羽毛球视频智能剪辑系统'
        if 'allow_register' in data:
            current['allow_register'] = bool(data['allow_register'])
        if 'max_upload_size_mb' in data:
            size_mb = int(data['max_upload_size_mb'])
            if size_mb < 1 or size_mb > 10240:
                return jsonify({'success': False, 'error': '上传大小必须在1-10240MB之间'}), 400
            current['max_upload_size_mb'] = size_mb
        if 'allowed_video_formats' in data:
            formats = data['allowed_video_formats']
            if isinstance(formats, str):
                formats = [f.strip() for f in formats.split(',') if f.strip()]
            if not formats or len(formats) == 0:
                return jsonify({'success': False, 'error': '至少指定一种视频格式'}), 400
            current['allowed_video_formats'] = [f.lower() for f in formats]
        
        # 更新邮件设置
        if 'email' in data and isinstance(data['email'], dict):
            email_cfg = data['email']
            current['email']['enabled'] = bool(email_cfg.get('enabled', False))
            current['email']['smtp_server'] = str(email_cfg.get('smtp_server', 'smtp.gmail.com')).strip()
            port = int(email_cfg.get('smtp_port', 587))
            if port < 1 or port > 65535:
                return jsonify({'success': False, 'error': 'SMTP端口不合法'}), 400
            current['email']['smtp_port'] = port
            current['email']['use_tls'] = bool(email_cfg.get('use_tls', True))
            current['email']['sender_email'] = str(email_cfg.get('sender_email', '')).strip()
            current['email']['sender_name'] = str(email_cfg.get('sender_name', '羽毛球视频剪辑系统')).strip()
            pwd = email_cfg.get('sender_password', '')
            # 如果传来的不是占位符，才更新密码
            if pwd and pwd != '********':
                current['email']['sender_password'] = pwd
        
        save_system_settings(current)
        
        # 同步更新邮件服务的内存配置
        email_service.config = current.get('email', {})
        # 同时保存到 email_config.json 保持兼容
        email_config_path = PROJECT_ROOT / 'config' / 'email_config.json'
        try:
            email_cfg_to_save = dict(current['email'])
            email_cfg_to_save['use_env_vars'] = False
            with open(email_config_path, 'w', encoding='utf-8') as f:
                json.dump(email_cfg_to_save, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        
        log_operation(session.get('user_id'), 'update_settings', '管理员修改了系统设置')
        
        return jsonify({'success': True, 'message': '设置已保存'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/api/settings/test-email', methods=['POST'])
@admin_required
def admin_test_email_settings():
    """测试当前邮件配置"""
    try:
        current = get_current_settings()
        email_cfg = current.get('email', {})
        
        if not email_cfg.get('enabled'):
            return jsonify({'success': False, 'message': '邮件服务未启用'})
        
        if not email_cfg.get('sender_email') or not email_cfg.get('sender_password'):
            return jsonify({'success': False, 'message': '发件人邮箱或密码未配置'})
        
        # 临时替换邮件服务配置进行测试
        original_config = dict(email_service.config)
        email_service.config = email_cfg
        result = email_service.test_connection()
        email_service.config = original_config
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'测试异常: {str(e)}'}), 500

@app.route('/')
@login_required
def index():
    """主页 - 视频上传页面"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_video():
    """处理视频上传"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': '未选择文件'}), 400
        
        file = request.files['video']
        match_type = request.form.get('match_type', 'singles')
        
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 动态检查文件大小
        max_size = get_max_upload_size()
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > max_size:
            max_mb = max_size // (1024 * 1024)
            return jsonify({'error': f'文件大小超过系统限制（最大{max_mb}MB）'}), 400
        
        if file and allowed_file(file.filename):
            # 生成唯一文件名
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{unique_id}{ext}"
            
            # 保存文件
            filepath = os.path.join(UPLOAD_FOLDER, new_filename)
            file.save(filepath)
            
            # 获取视频信息
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # 保存到session
            session['current_video'] = {
                'filename': new_filename,
                'filepath': filepath,
                'match_type': match_type,
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'upload_time': datetime.now().isoformat()
            }
            
            # 记录上传统计
            from datetime import date
            user_id = session.get('user_id')
            if user_id:
                increment_user_stat(user_id, date.today(), 'videos_uploaded')
                increment_user_stat(user_id, date.today(), 'total_upload_size', file_size)
            
            return jsonify({
                'success': True,
                'filename': new_filename,
                'duration': duration,
                'match_type': match_type,
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'duration': duration
                }
            })
        
        return jsonify({'error': '不支持的文件格式'}), 400
        
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/detection')
@login_required
def detection():
    """智能检测页面"""
    if 'current_video' not in session:
        return redirect('/')

    video_info = session['current_video']

    return render_template('detection.html',
                         video_info=video_info)

@app.route('/start_detection', methods=['POST'])
@login_required
def start_detection():
    """开始智能检测 - 支持并发，每个用户独立任务"""
    try:
        if 'current_video' not in session:
            return jsonify({'error': '未找到视频'}), 400

        # 清理旧任务
        _cleanup_old_tasks()

        # 检查当前运行中的任务数
        with detection_tasks_lock:
            running_count = sum(
                1 for t in detection_tasks.values()
                if t['status'] == 'running' and t.get('thread') and t['thread'].is_alive()
            )
            if running_count >= MAX_CONCURRENT_TASKS:
                return jsonify({'error': f'当前并发检测任务已达上限({MAX_CONCURRENT_TASKS})，请稍后重试'}), 429

        video_info = session['current_video']
        video_info['user_id'] = session.get('user_id')

        # 寻找最新的最优模型
        model_dir = PROJECT_ROOT / '03_model' / 'trained'
        model_dir_str = str(model_dir)
        best_models = sorted([f for f in os.listdir(model_dir_str) if f.startswith('best_model_')])
        if not best_models:
            return jsonify({'error': '未找到训练好的模型权重，请先进行模型训练'}), 500

        model_path = str(model_dir / best_models[-1])
        print(f"使用模型进行Web预测: {best_models[-1]}")

        # 创建并发检测任务
        task_id = _create_task(video_info, model_path)

        # 将 task_id 存入 session，后续轮询/获取结果时使用
        session['detection_task_id'] = task_id

        return jsonify({
            'success': True,
            'message': '检测已启动',
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'error': f'启动检测失败: {str(e)}'}), 500


@app.route('/get_detection_progress', methods=['GET'])
@login_required
def get_detection_progress():
    """获取检测进度 - 供前端轮询，按 task_id 隔离"""
    task_id = session.get('detection_task_id')
    if not task_id:
        return jsonify({
            'status': 'idle',
            'progress': 0,
            'message': '',
            'result': None,
            'error': None
        })

    with detection_tasks_lock:
        task = detection_tasks.get(task_id)
        if not task:
            return jsonify({
                'status': 'idle',
                'progress': 0,
                'message': '',
                'result': None,
                'error': '任务不存在或已过期'
            })

        # 检测僵尸状态（status=running 但线程已死）
        if task['status'] == 'running' and task.get('thread') and not task['thread'].is_alive():
            task['status'] = 'error'
            task['error'] = '检测线程异常终止'

        return jsonify({
            'status': task['status'],
            'progress': task['progress'],
            'message': task['message'],
            'result': task['result'],
            'error': task['error']
        })

@app.route('/player')
@login_required
def player():
    """回合播放器页面"""
    if 'current_video' not in session:
        return redirect('/')

    video_info = session['current_video']

    # 优先从 session 取检测结果；若没有则尝试从任务存储获取
    if 'detection_results' not in session:
        task_id = session.get('detection_task_id')
        if task_id:
            with detection_tasks_lock:
                task = detection_tasks.get(task_id)
                if task and task.get('status') == 'completed' and task.get('result'):
                    session['detection_results'] = task['result']
                else:
                    return redirect('/')  # 没有检测结果，回首页
        else:
            return redirect('/')

    detection_results = session['detection_results']

    # 预处理回合数据：优先使用 all_raw_rounds（含被过滤的回合，供用户自主选择）
    raw_rounds = detection_results.get('all_raw_rounds', [])
    if not raw_rounds:
        raw_rounds = detection_results.get('detected_rounds', [])
    auto_approved_ids = detection_results.get('auto_approved_ids', [])

    return render_template('player.html',
                         video_info=video_info,
                         detection_results=detection_results,
                         display_rounds=raw_rounds,
                         auto_approved_ids=auto_approved_ids)

@app.route('/get_video/<filename>')
@login_required
def get_video(filename):
    """获取视频文件（支持范围请求，用于大视频流式播放）"""
    try:
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(video_path):
            return jsonify({'error': '视频文件不存在'}), 404
        
        # 获取文件大小
        file_size = os.path.getsize(video_path)
        
        # 处理范围请求（支持视频拖动进度条）
        range_header = request.headers.get('Range', None)
        
        if range_header:
            # 解析范围请求
            start = 0
            end = file_size - 1
            
            # 格式: bytes=start-end
            if range_header.startswith('bytes='):
                range_value = range_header[6:]
                if '-' in range_value:
                    parts = range_value.split('-')
                    if parts[0]:
                        start = int(parts[0])
                    if parts[1]:
                        end = int(parts[1])
            
            # 确保范围有效
            if start >= file_size or end >= file_size:
                return jsonify({'error': '无效的范围请求'}), 416
            
            # 读取指定范围的数据
            with open(video_path, 'rb') as f:
                f.seek(start)
                data = f.read(end - start + 1)
            
            # 返回部分内容
            response = Flask.response_class(
                response=data,
                status=206,
                mimetype='video/mp4',
                direct_passthrough=True
            )
            response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Content-Length', str(len(data)))
            return response
        else:
            # 普通请求，返回整个文件
            return send_file(video_path, mimetype='video/mp4')
            
    except Exception as e:
        return jsonify({'error': f'获取视频失败: {str(e)}'}), 500

@app.route('/download_round', methods=['POST'])
@login_required
def download_round():
    """下载指定回合的视频片段"""
    try:
        if 'current_video' not in session:
            return jsonify({'error': '未找到视频'}), 400
        
        data = request.json
        round_id = data.get('round_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        video_info = session['current_video']
        input_path = video_info['filepath']
        
        # 生成输出文件名
        output_filename = f"round_{round_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # 检查FFmpeg是否可用
        try:
            import subprocess
            
            # 测试FFmpeg是否安装
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                raise FileNotFoundError("FFmpeg not found")
            
            # 使用FFmpeg剪辑视频
            cmd = [
                'ffmpeg', '-i', input_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'filename': output_filename,
                    'download_url': f'/download_file/{output_filename}'
                })
            else:
                return jsonify({'error': f'视频剪辑失败: {result.stderr}'}), 500
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # FFmpeg不可用时的备选方案
            return jsonify({
                'error': 'FFmpeg未安装或不可用。请安装FFmpeg后重试。\n下载地址: https://ffmpeg.org/download.html'
            }), 500
        
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/download_file/<filename>')
@login_required
def download_file(filename):
    """下载处理后的文件"""
    try:
        file_path = os.path.join(PROCESSED_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """获取当前状态"""
    return jsonify({
        'has_video': 'current_video' in session,
        'has_detection': 'detection_results' in session,
        'current_step': get_current_step()
    })

@app.route('/api/confirm_rounds', methods=['POST'])
@login_required
def api_confirm_rounds():
    """保存用户手动选择的回合"""
    try:
        data = request.json
        selected_ids = data.get('selected_ids', [])
        
        # 优先从 session 取检测结果；若没有则从任务存储获取
        detection_results = session.get('detection_results')
        if detection_results is None:
            task_id = session.get('detection_task_id')
            if task_id:
                with detection_tasks_lock:
                    task = detection_tasks.get(task_id)
                    if task and task.get('status') == 'completed' and task.get('result'):
                        detection_results = task['result']
                        session['detection_results'] = detection_results
            if detection_results is None:
                return jsonify({'success': False, 'error': '未找到检测结果'}), 400
        
        all_raw_rounds = detection_results.get('all_raw_rounds', [])
        
        # 按用户选择过滤回合
        selected_rounds = [r for r in all_raw_rounds if r['round_id'] in selected_ids]
        
        # 去掉内部标记字段
        for r in selected_rounds:
            r.pop('_auto_filtered', None)
            r.pop('_filter_reason', None)
        
        # 重新编号round_id
        for i, r in enumerate(selected_rounds, 1):
            r['round_id'] = i
        
        # 更新检测结果
        if selected_rounds:
            detection_results['detected_rounds'] = selected_rounds
            detection_results['statistics'] = {
                'total_rounds': len(selected_rounds),
                'total_active_time': sum(r['duration'] for r in selected_rounds),
                'average_round_duration': sum(r['duration'] for r in selected_rounds) / len(selected_rounds),
                'detection_accuracy': 1.0
            }
        else:
            detection_results['detected_rounds'] = []
            detection_results['statistics'] = {'total_rounds': 0}
        
        session['detection_results'] = detection_results
        
        return jsonify({
            'success': True,
            'round_count': len(selected_rounds),
            'total_active_time': round(detection_results['statistics'].get('total_active_time', 0), 2)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def get_current_step():
    """获取当前步骤"""
    if 'detection_results' in session:
        return 'player'
    elif 'current_video' in session:
        return 'detection'
    else:
        return 'upload'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)