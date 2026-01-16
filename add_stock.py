import os
import re
import time
import requests

def get_telegram_updates(bot_token):
    """获取 Telegram 机器人最近收到的消息"""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        # timeout=10 避免卡死
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        print(f"获取消息失败: {e}")
    return []

def send_reply(bot_token, chat_id, text):
    """发送回复消息"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=data, timeout=5)
    except:
        pass

def main():
    bot_token = os.getenv("TG_BOT_TOKEN")
    admin_chat_id = os.getenv("TG_CHAT_ID")  # 只处理你发的消息，防止别人乱加

    if not bot_token:
        print("未设置 TG_BOT_TOKEN")
        return

    # 1. 获取消息
    updates = get_telegram_updates(bot_token)
    if not updates:
        print("没有新消息")
        return

    # 2. 读取现有股票列表
    file_path = "stock_list.txt"
    existing_stocks = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_stocks = {line.strip() for line in f if line.strip()}

    new_stocks = set()
    latest_update_id = 0

    # 3. 解析消息
    # 我们只处理最近 40 分钟内的消息 (防止重复处理太久以前的)
    current_time = time.time()
    
    for update in updates:
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "")
        date = message.get("date", 0)
        update_id = update.get("update_id")

        latest_update_id = max(latest_update_id, update_id)

        # 安全检查：只处理你的 Chat ID 发出的消息
        if admin_chat_id and chat_id != str(admin_chat_id):
            continue

        # 时间检查：忽略 40 分钟以前的消息 (GitHub Action 每30分钟跑一次，留10分钟冗余)
        if current_time - date > 2400: 
            continue

        # 正则提取：匹配 6 位数字代码 (如 600970, 000001)
        # \b 确保是独立的数字，不是电话号码的一部分
        codes = re.findall(r"\b\d{6}\b", text)
        
        for code in codes:
            if code not in existing_stocks:
                new_stocks.add(code)
                print(f"发现新股票代码: {code}")

    # 4. 如果有新股票，写入文件
    if new_stocks:
        # 更新集合
        final_list = existing_stocks.union(new_stocks)
        
        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            for stock in sorted(final_list):
                f.write(f"{stock}\n")
        
        print(f"已保存 {len(new_stocks)} 只新股票到列表。")
        
        # (可选) 消费掉消息，防止下次还读到
        # Telegram 通过 offset 确认消息已读
        requests.get(f"https://api.telegram.org/bot{bot_token}/getUpdates?offset={latest_update_id + 1}")

        # 给 TG 发送确认通知
        msg = f"✅ <b>已添加监控:</b>\n" + "\n".join(new_stocks)
        send_reply(bot_token, admin_chat_id, msg)
        
    else:
        print("没有检测到有效的新股票代码。")

if __name__ == "__main__":
    main()
