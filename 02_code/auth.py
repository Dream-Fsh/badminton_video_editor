"""
用户认证模块
提供用户注册、登录、登出功能
"""

import sqlite3
import hashlib
import secrets
import random
import string
from datetime import datetime
from pathlib import Path

def generate_verification_code(length=6):
    """生成数字验证码"""
    return ''.join(random.choices(string.digits, k=length))

def store_verification_code(email, code, expires_minutes=10):
    """存储验证码到数据库"""
    from datetime import datetime, timedelta
    conn = get_db()
    cursor = conn.cursor()

    expires_at = datetime.now() + timedelta(minutes=expires_minutes)

    # 清除该邮箱未使用的旧验证码
    cursor.execute('''
        DELETE FROM email_verification_codes
        WHERE email = ? AND used = 0
    ''', (email,))

    cursor.execute('''
        INSERT INTO email_verification_codes (email, verification_code, expires_at)
        VALUES (?, ?, ?)
    ''', (email, code, expires_at.isoformat()))

    conn.commit()
    conn.close()

def verify_code(email, code):
    """验证验证码"""
    from datetime import datetime
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, verification_code, expires_at
        FROM email_verification_codes
        WHERE email = ? AND used = 0
        ORDER BY created_at DESC
        LIMIT 1
    ''', (email,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return False

    expires_at = datetime.fromisoformat(row['expires_at'])
    if datetime.now() > expires_at:
        conn.close()
        return False

    if row['verification_code'] != code:
        conn.close()
        return False

    # 标记为已使用
    cursor.execute('''
        UPDATE email_verification_codes
        SET used = 1, used_at = ?
        WHERE id = ?
    ''', (datetime.now().isoformat(), row['id']))

    conn.commit()
    conn.close()
    return True

def get_verification_code(email):
    """获取验证码（用于测试/开发模式）"""
    from datetime import datetime
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT verification_code, expires_at
        FROM email_verification_codes
        WHERE email = ? AND used = 0
        ORDER BY created_at DESC
        LIMIT 1
    ''', (email,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    expires_at = datetime.fromisoformat(row['expires_at'])
    if datetime.now() > expires_at:
        return None

    return row['verification_code']

# 数据库路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DB_PATH = PROJECT_ROOT / 'data' / 'users.db'

def get_db():
    """获取数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表"""
    conn = get_db()
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # 创建操作日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 创建视频表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            duration REAL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建邮件验证码表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_verification_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            verification_code TEXT NOT NULL,
            code_type TEXT NOT NULL DEFAULT 'reset_password',
            used INTEGER NOT NULL DEFAULT 0,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            used_at TIMESTAMP
        )
    ''')

    # 创建用户统计数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date DATE NOT NULL,
            videos_uploaded INTEGER DEFAULT 0,
            videos_processed INTEGER DEFAULT 0,
            total_upload_size INTEGER DEFAULT 0,
            total_processing_time INTEGER DEFAULT 0,
            login_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, date)
        )
    ''')

    conn.commit()
    conn.close()
    
    # 创建默认管理员账户（如果不存在）
    create_default_admin()

def create_default_admin():
    """创建默认管理员账户"""
    conn = get_db()
    cursor = conn.cursor()
    
    # 检查是否已存在管理员
    cursor.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
    if cursor.fetchone() is None:
        # 创建默认管理员: admin / admin123
        admin_password = hash_password('admin123')
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role)
            VALUES (?, ?, ?, ?)
        ''', ('admin', admin_password, 'admin@badminton.local', 'admin'))
        conn.commit()
        print("默认管理员账户已创建: admin / admin123")
    
    conn.close()

def hash_password(password, salt=None):
    """密码哈希（使用SHA-256 + salt）"""
    if salt is None:
        salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((salt + password).encode())
    return f"{salt}${hash_obj.hexdigest()}"

def verify_password(password, stored_hash):
    """验证密码"""
    try:
        salt, _ = stored_hash.split('$')
        return hash_password(password, salt) == stored_hash
    except:
        return False

def register_user(username, password, email=None):
    """注册新用户"""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return {'success': False, 'error': '用户名已存在'}
        
        # 创建用户
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role)
            VALUES (?, ?, ?, ?)
        ''', (username, password_hash, email, 'user'))
        
        conn.commit()
        user_id = cursor.lastrowid
        
        # 记录日志
        log_operation(user_id, 'register', f'用户注册: {username}')
        
        conn.close()
        return {'success': True, 'user_id': user_id, 'message': '注册成功'}
        
    except Exception as e:
        conn.close()
        return {'success': False, 'error': str(e)}

def authenticate_user(username, password):
    """验证用户登录"""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, username, password_hash, role, is_active
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return {'success': False, 'error': '用户名或密码错误'}
        
        if not user['is_active']:
            conn.close()
            return {'success': False, 'error': '账户已被禁用'}
        
        if not verify_password(password, user['password_hash']):
            conn.close()
            return {'success': False, 'error': '用户名或密码错误'}
        
        # 更新最后登录时间
        cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                      (datetime.now().isoformat(), user['id']))
        conn.commit()
        
        # 记录日志
        log_operation(user['id'], 'login', f'用户登录: {username}')
        
        conn.close()
        return {
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role']
            }
        }
        
    except Exception as e:
        conn.close()
        return {'success': False, 'error': str(e)}

def log_operation(user_id, action, details, ip_address=None):
    """记录操作日志"""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO operation_logs (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action, details, ip_address))
        conn.commit()
    except:
        pass
    
    conn.close()

def get_all_users():
    """获取所有用户（管理员功能）"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, role, created_at, last_login, is_active
        FROM users ORDER BY created_at DESC
    ''')
    
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return users

def get_user_by_id(user_id):
    """根据ID获取用户"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, role, created_at, last_login, is_active
        FROM users WHERE id = ?
    ''', (user_id,))
    
    user = cursor.fetchone()
    conn.close()
    
    return dict(user) if user else None

def update_user_status(user_id, is_active):
    """更新用户状态"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (is_active, user_id))
    conn.commit()
    conn.close()
    
    return True

def reset_user_password(user_id, new_password):
    """重置用户密码（管理员功能）"""
    conn = get_db()
    cursor = conn.cursor()
    
    password_hash = hash_password(new_password)
    cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user_id))
    conn.commit()
    conn.close()
    
    return True

def find_user_by_username(username):
    """根据用户名查找用户"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username, email, role FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    return dict(user) if user else None

def get_user_email(username):
    """获取用户邮箱"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT email FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    
    return result['email'] if result and result['email'] else None

def delete_user(user_id):
    """删除用户"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    return True

def get_operation_logs(limit=50):
    """获取操作日志"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT l.*, u.username
        FROM operation_logs l
        LEFT JOIN users u ON l.user_id = u.id
        ORDER BY l.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return logs

def get_user_count():
    """获取用户总数"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM users')
    count = cursor.fetchone()['count']
    conn.close()
    
    return count


# ==================== videos 表操作 ====================

def create_video(filename, file_path, duration=None):
    """创建视频记录"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO videos (filename, file_path, duration)
        VALUES (?, ?, ?)
    ''', (filename, file_path, duration))
    
    conn.commit()
    video_id = cursor.lastrowid
    conn.close()
    
    return video_id

def get_all_videos():
    """获取所有视频"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM videos ORDER BY created_at DESC')
    videos = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return videos

def get_video_by_id(video_id):
    """根据ID获取视频"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
    video = cursor.fetchone()
    conn.close()
    
    return dict(video) if video else None

def update_video_status(video_id, status):
    """更新视频状态"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('UPDATE videos SET status = ? WHERE id = ?', (status, video_id))
    conn.commit()
    conn.close()
    
    return True

def delete_video(video_id):
    """删除视频记录"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM videos WHERE id = ?', (video_id,))
    conn.commit()
    conn.close()
    
    return True

def get_video_count():
    """获取视频总数"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM videos')
    count = cursor.fetchone()['count']
    conn.close()
    
    return count


# ==================== user_statistics 表操作 ====================

def get_or_create_user_statistics(user_id, date):
    """获取或创建用户当日统计记录"""
    from datetime import date as dt_date
    if isinstance(date, dt_date):
        date = date.isoformat()
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM user_statistics
        WHERE user_id = ? AND date = ?
    ''', (user_id, date))
    
    row = cursor.fetchone()
    if row:
        conn.close()
        return dict(row)
    
    cursor.execute('''
        INSERT INTO user_statistics (user_id, date)
        VALUES (?, ?)
    ''', (user_id, date))
    
    conn.commit()
    stat_id = cursor.lastrowid
    
    cursor.execute('SELECT * FROM user_statistics WHERE id = ?', (stat_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row)

def increment_user_stat(user_id, date, field, value=1):
    """增加用户统计字段值"""
    from datetime import date as dt_date, datetime
    if isinstance(date, dt_date):
        date = date.isoformat()
    
    allowed_fields = ['videos_uploaded', 'videos_processed', 'total_upload_size',
                      'total_processing_time', 'login_count']
    if field not in allowed_fields:
        return False
    
    conn = get_db()
    cursor = conn.cursor()
    
    # 先确保记录存在
    cursor.execute('''
        INSERT OR IGNORE INTO user_statistics (user_id, date)
        VALUES (?, ?)
    ''', (user_id, date))
    
    # 更新指定字段
    cursor.execute(f'''
        UPDATE user_statistics
        SET {field} = {field} + ?, updated_at = ?
        WHERE user_id = ? AND date = ?
    ''', (value, datetime.now().isoformat(), user_id, date))
    
    conn.commit()
    conn.close()
    
    return True

def get_user_statistics(user_id, date=None):
    """获取用户统计"""
    from datetime import date as dt_date
    conn = get_db()
    cursor = conn.cursor()
    
    if date:
        if isinstance(date, dt_date):
            date = date.isoformat()
        cursor.execute('''
            SELECT * FROM user_statistics
            WHERE user_id = ? AND date = ?
        ''', (user_id, date))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    else:
        cursor.execute('''
            SELECT * FROM user_statistics
            WHERE user_id = ?
            ORDER BY date DESC
        ''', (user_id,))
        stats = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return stats


# 初始化数据库
init_db()
