"""
邮件服务模块
提供邮件发送功能，支持忘记密码验证码发送
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def load_email_config():
    """加载邮件配置"""
    config_path = PROJECT_ROOT / 'config' / 'email_config.json'
    
    # 默认配置（开发模式）
    default_config = {
        "enabled": False,  # 默认关闭，使用开发模式
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
        "sender_email": "",
        "sender_password": "",
        "sender_name": "羽毛球视频剪辑系统",
        "use_env_vars": True  # 优先使用环境变量
    }
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            # 合并配置
            for key, value in user_config.items():
                if not key.startswith('_'):
                    default_config[key] = value
    
    # 检查环境变量（优先级最高）
    import os
    if default_config.get('use_env_vars', True):
        env_sender = os.environ.get('EMAIL_SENDER')
        env_password = os.environ.get('EMAIL_PASSWORD')
        if env_sender:
            default_config['sender_email'] = env_sender
        if env_password:
            default_config['sender_password'] = env_password
    
    return default_config

class EmailService:
    """邮件服务类"""
    
    def __init__(self):
        self.config = load_email_config()
    
    @property
    def is_enabled(self):
        """检查邮件服务是否启用"""
        return self.config.get('enabled', False) and bool(self.config.get('sender_email'))
    
    def send_verification_code(self, to_email, username, code):
        """
        发送验证码邮件
        
        Args:
            to_email: 收件人邮箱
            username: 用户名
            code: 验证码
        
        Returns:
            bool: 发送是否成功
        """
        if not self.is_enabled:
            print(f"[邮件服务] 邮件服务未启用，验证码将显示在页面上")
            return False
        
        try:
            # 构建邮件内容
            subject = f"【{self.config.get('sender_name', '系统')}】密码重置验证码"
            
            html_content = self._get_html_template(username, code)
            text_content = self._get_text_template(username, code)
            
            # 创建邮件对象
            from email.header import Header
            message = MIMEMultipart("alternative")
            message["Subject"] = Header(subject, 'utf-8')
            message["From"] = f"{self.config['sender_email']}"
            message["To"] = to_email
            
            # 添加HTML和纯文本版本
            message.attach(MIMEText(text_content, "plain", "utf-8"))
            message.attach(MIMEText(html_content, "html", "utf-8"))
            
            # 发送邮件
            return self._send_email(message, to_email)
            
        except Exception as e:
            print(f"[邮件服务] 发送邮件失败: {str(e)}")
            return False
    
    def _get_html_template(self, username, code):
        """获取HTML邮件模板"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .content {{ padding: 30px; }}
                .code-box {{ background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0; }}
                .code {{ font-size: 32px; font-weight: bold; color: #667eea; letter-spacing: 8px; }}
                .warning {{ color: #dc3545; font-size: 14px; margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 6px; }}
                .footer {{ text-align: center; padding: 20px; color: #6c757d; font-size: 12px; border-top: 1px solid #dee2e6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>密码重置验证码</h1>
                </div>
                <div class="content">
                    <p>尊敬的用户 <strong>{username}</strong>：</p>
                    <p>您正在申请重置密码，请使用以下验证码完成验证：</p>
                    <div class="code-box">
                        <div class="code">{code}</div>
                    </div>
                    <p>验证码有效期：<strong>10分钟</strong></p>
                    <div class="warning">
                        <strong>安全提示：</strong><br>
                        • 请勿将验证码告知他人<br>
                        • 如果这不是您的操作，请忽略此邮件<br>
                        • 本邮件由系统自动发送，请勿回复
                    </div>
                </div>
                <div class="footer">
                    <p>此邮件由系统自动发送于 {timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_text_template(self, username, code):
        """获取纯文本邮件模板"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""
        尊敬的用户 {username}：

        您正在申请重置密码，请使用以下验证码完成验证：

        验证码：{code}

        验证码有效期：10分钟

        安全提示：
        • 请勿将验证码告知他人
        • 如果这不是您的操作，请忽略此邮件
        • 本邮件由系统自动发送，请勿回复

        此邮件由系统自动发送于 {timestamp}
        """
    
    def _send_email(self, message, to_email):
        """实际发送邮件"""
        try:
            server = self.config['smtp_server']
            port = self.config['smtp_port']
            use_tls = self.config.get('use_tls', True)
            sender = self.config['sender_email']
            password = self.config['sender_password']
            
            # 根据端口选择连接方式
            if use_tls and port == 587:
                # 使用 STARTTLS
                context = ssl.create_default_context()
                with smtplib.SMTP(server, port) as server:
                    server.starttls(context=context)
                    server.login(sender, password)
                    server.sendmail(sender, to_email, message.as_string())
            elif port == 465:
                # 使用 SSL
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(server, port, context=context) as server:
                    server.login(sender, password)
                    server.sendmail(sender, to_email, message.as_string())
            else:
                # 普通连接（不推荐）
                with smtplib.SMTP(server, port) as server:
                    server.login(sender, password)
                    server.sendmail(sender, to_email, message.as_string())
            
            print(f"[邮件服务] 邮件发送成功 -> {to_email}")
            return True
            
        except Exception as e:
            print(f"[邮件服务] SMTP发送失败: {str(e)}")
            return False
    
    def test_connection(self):
        """测试邮件连接"""
        if not self.is_enabled:
            return {"success": False, "message": "邮件服务未启用"}
        
        try:
            server = self.config['smtp_server']
            port = self.config['smtp_port']
            use_tls = self.config.get('use_tls', True)
            sender = self.config['sender_email']
            password = self.config['sender_password']
            
            if use_tls and port == 587:
                context = ssl.create_default_context()
                with smtplib.SMTP(server, port, timeout=10) as server:
                    server.starttls(context=context)
                    server.login(sender, password)
            elif port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(server, port, context=context, timeout=10) as server:
                    server.login(sender, password)
            else:
                with smtplib.SMTP(server, port, timeout=10) as server:
                    server.login(sender, password)
            
            return {"success": True, "message": "邮件服务连接正常"}
            
        except Exception as e:
            return {"success": False, "message": f"连接失败: {str(e)}"}

# 全局邮件服务实例
email_service = EmailService()

def send_password_reset_email(to_email, username, code):
    """发送密码重置邮件的便捷函数"""
    return email_service.send_verification_code(to_email, username, code)

def test_email_service():
    """测试邮件服务（供管理员使用）"""
    return email_service.test_connection()
