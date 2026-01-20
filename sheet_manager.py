import os
import json
import gspread
from google.oauth2.service_account import Credentials

class SheetManager:
    def __init__(self):
        # 1. 获取凭证
        raw_key = os.getenv("GCP_SA_KEY")
        if not raw_key:
            raise ValueError("❌ 环境变量 GCP_SA_KEY 未找到")
        
        try:
            creds_dict = json.loads(raw_key)
            creds = Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            )
        except json.JSONDecodeError:
            raise ValueError("❌ GCP_SA_KEY JSON 解析失败，请检查格式")

        # 2. 连接客户端
        print("   >>> [System] 初始化 Google Sheets (智能连接版)...")
        try:
            self.client = gspread.authorize(creds)
            print("   ✅ Google Auth 认证成功")
            print(f"   🤖 当前机器人: {creds.service_account_email}")
        except Exception as e:
            raise Exception(f"❌ Google Auth 失败: {e}")

        # 3. 连接表格 (优先 ID，后文件名)
        sheet_name_or_id = os.getenv("SHEET_NAME")
        if not sheet_name_or_id:
            raise ValueError("❌ 环境变量 SHEET_NAME 未找到")

        try:
            # 尝试按 ID 打开 (如果是长字符串)
            if len(sheet_name_or_id) > 20: 
                self.sh = self.client.open_by_key(sheet_name_or_id)
                print(f"   ✅ [成功] 已通过 ID 连接到表格！")
            else:
                print(f"   >>> 正在尝试按【文件名】打开: '{sheet_name_or_id}'...")
                self.sh = self.client.open(sheet_name_or_id)
                print(f"   ✅ [成功] 已通过文件名连接到表格！")
        except gspread.SpreadsheetNotFound:
            print(f"   ❌ 找不到名为 '{sheet_name_or_id}' 的表格。")
            print("   ⚠️ 请确保表格已分享给机器人邮箱 (见上文)")
            raise

        # 默认操作第一个工作表
        self.sheet = self.sh.sheet1

    def get_all_stocks(self):
        """
        获取所有股票配置，返回字典格式
        Format: {'000001': {'date': '2023-01-01', 'price': 10.5, 'qty': 100}, ...}
        """
        all_values = self.sheet.get_all_values()
        if not all_values:
            return {}
        
        # 跳过表头 (假设第一行是 Code, BuyDate, Price, Qty)
        headers = all_values[0]
        data_rows = all_values[1:]
        
        stocks = {}
        for row in data_rows:
            if not row or not row[0].strip(): continue
            
            # 强制补全6位代码
            raw_symbol = row[0].strip()
            digits = ''.join(filter(str.isdigit, raw_symbol))
            symbol = digits.zfill(6)
            
            # 安全获取其他列
            buy_date = row[1].strip() if len(row) > 1 else ""
            price = row[2].strip() if len(row) > 2 else ""
            qty = row[3].strip() if len(row) > 3 else ""
            
            stocks[symbol] = {
                "date": buy_date,
                "price": price,
                "qty": qty
            }
        return stocks

    def add_or_update_stock(self, symbol, date='', price='', qty=''):
        """
        添加或更新股票 (修复了 NoneType 和 CellNotFound 错误)
        """
        # 1. 格式化代码
        clean_symbol = ''.join(filter(str.isdigit, str(symbol))).zfill(6)
        print(f"   🔍 正在查找股票: {clean_symbol}")
        
        try:
            # 2. 查找是否存在 (find 返回 Cell 对象或 None)
            cell = self.sheet.find(clean_symbol)
            
            if cell:
                # === 更新逻辑 ===
                print(f"   Found at Row {cell.row}. Updating...")
                row = cell.row
                # 如果提供了新值，才更新对应列
                # 假设列顺序: A=Code(1), B=Date(2), C=Price(3), D=Qty(4)
                if date: 
                    self.sheet.update_cell(row, 2, str(date))
                if price: 
                    self.sheet.update_cell(row, 3, str(price))
                if qty: 
                    self.sheet.update_cell(row, 4, str(qty))
                return f"✅ 已更新 {clean_symbol}"
            
            else:
                # === 新增逻辑 ===
                print(f"   Not found. Appending new row...")
                # 追加一行: [Code, Date, Price, Qty]
                self.sheet.append_row([clean_symbol, str(date), str(price), str(qty)])
                return f"🆕 已添加关注 {clean_symbol}"
                
        except Exception as e:
            print(f"   ❌ 操作表格失败: {e}")
            # 抛出异常以便上层捕获
            raise e
