import os
import re
import requests
from sheet_manager import SheetManager

def get_telegram_updates(bot_token, offset=None):
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    params = {"timeout": 10}
    if offset:
        params["offset"] = offset
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        print(f"   ⚠️ 获取 Telegram 消息失败: {e}")
    return []

def send_telegram_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        requests.post(url, json=data, timeout=10)
    except:
        pass

def parse_command(text):
    """
    解析指令，提取：意图(add/del), 代码, 日期, 价格, 数量
    """
    text = text.strip()
    
    # 1. 提取股票代码 (6位数字)
    code_match = re.search(r"\d{6}", text)
    if not code_match:
        return None
    code = code_match.group()
    
    # 2. 判断意图
    intent = "add"
    if any(k in text for k in ["删除", "移除", "del", "remove", "取消"]):
        intent = "remove"
    
    # 3. 提取其他参数 (日期、数字)
    # 移除掉代码和关键词，剩下的部分尝试解析
    remain_text = text.replace(code, "").replace("关注", "").replace("add", "")
    
    # 提取日期 (YYYY-MM-DD 或 YYYY/MM/DD)
    date = ""
    date_match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", remain_text)
    if date_match:
        date = date_match.group()
        remain_text = remain_text.replace(date, "") # 移除已识别的日期
    
    # 提取剩下的数字 (价格、数量)
    # 简单的按顺序：第一个浮点数是价格，第二个是数量
    nums = re.findall(r"\d+\.?\d*", remain_text)
    price = ""
    qty = ""
    
    if len(nums) >= 1: price = nums[0]
    if len(nums) >= 2: qty = nums[1]
    
    return {
        "intent": intent,
        "code": code,
        "date": date,
        "price": price,
        "qty": qty
    }

def main():
    bot_token = os.getenv("TG_BOT_TOKEN")
    if not bot_token:
        print("❌ 缺少 TG_BOT_TOKEN")
        return

    print("☁️ 正在连接 Google Sheets...")
    try:
        sm = SheetManager()
        print("✅ 表格连接成功")
    except Exception as e:
        print(f"❌ 表格连接失败: {e}")
        return

    # 获取消息 (这里简化逻辑，实际生产中可能需要记录 offset 避免重复处理)
    # 在 GitHub Actions 每次运行通常处理最新的一批
    updates = get_telegram_updates(bot_token)
    
    # 如果没有消息，直接退出
    if not updates:
        print("📭 无新消息")
        return

    print(f"📥 收到 {len(updates)} 条消息，开始处理...")
    
    # 只需要处理最新的几条，或者全部处理
    # 为了避免死循环，这里假设 GitHub Actions 频率较低，
    # 或者你需要一个机制来标记已读 (offset)。
    # 简单起见，我们处理完消息后，不更新 offset，依赖 Telegram 的保留时长(24h)。
    # 但这会导致重复处理。
    # **优化**：我们只处理最近 10 分钟内的消息？或者简单处理所有 pending 的。
    # 为了防止 GitHub Actions 重复跑，建议在 `getUpdates` 后调用一次 `getUpdates` 带上最新的 `update_id + 1` 来清除队列。
    
    max_update_id = 0
    
    for update in updates:
        update_id = update["update_id"]
        if update_id > max_update_id:
            max_update_id = update_id
            
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")
        
        if not text or not chat_id: continue
        
        print(f"  -- 处理消息: {text}")
        
        parsed = parse_command(text)
        if not parsed:
            print("     -> 忽略 (非指令)")
            continue
            
        result_msg = ""
        
        if parsed["intent"] == "remove":
            result_msg = sm.remove_stock(parsed["code"])
        else:
            # Add or Update
            try:
                result_msg = sm.add_or_update_stock(
                    parsed["code"], 
                    parsed["date"], 
                    parsed["price"], 
                    parsed["qty"]
                )
            except Exception as e:
                result_msg = f"❌ 添加失败: {e}"
        
        print(f"     -> 结果: {result_msg}")
        # 发送回执
        send_telegram_message(bot_token, chat_id, result_msg)

    # 清除已处理的消息 (防止下次运行重复处理)
    if max_update_id > 0:
        print(f"🧹 清理消息队列 (Offset: {max_update_id + 1})")
        get_telegram_updates(bot_token, offset=max_update_id + 1)

if __name__ == "__main__":
    main()
