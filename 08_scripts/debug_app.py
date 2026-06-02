#!/usr/bin/env python3
"""
调试版本的Web应用 - 用于排查问题
"""

from flask import Flask, render_template, jsonify

app = Flask(__name__)
app.secret_key = 'debug_key'

@app.route('/')
def index():
    """主页"""
    try:
        return render_template('upload.html')
    except Exception as e:
        return f"模板渲染错误: {str(e)}"

@app.route('/test')
def test():
    """测试路由"""
    return jsonify({
        'status': 'ok',
        'message': 'Web应用运行正常'
    })

@app.route('/api/status')
def get_status():
    """获取状态"""
    return jsonify({
        'has_video': False,
        'has_detection': False,
        'current_step': 'upload'
    })

if __name__ == '__main__':
    print("启动调试版Web应用...")
    print("访问地址: http://localhost:5000")
    print("测试地址: http://localhost:5000/test")
    
    app.run(host='0.0.0.0', port=5000, debug=True)