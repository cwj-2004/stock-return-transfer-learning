"""
代码映射提取工具

功能：从 PDF 文件或文本文件提取北交所股票代码映射表（旧代码 -> 新代码）
输出：保存到 data/raw/对照.txt
"""

import os
import re
import warnings
from pathlib import Path


# === 配置路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_PATH = os.path.join(BASE_DIR, "data", "raw", "附件：北交所存量上市公司股票新旧代码对照表.pdf")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "对照.txt")


def extract_from_pdfplumber(pdf_path: str) -> list:
    """
    使用 pdfplumber 从 PDF 提取代码映射

    Args:
        pdf_path: PDF 文件路径

    Returns:
        代码映射列表 [(old_code, new_code), ...]
    """
    import pdfplumber

    rows = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF 页数: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages, 1):
            print(f"处理第 {i}/{len(pdf.pages)} 页...")

            try:
                # 尝试提取表格
                table = page.extract_table()
                if table:
                    for row in table:
                        if not row or all(x is None or str(x).strip() == "" for x in row):
                            continue
                        nums = [re.sub(r"\D", "", str(x) or "") for x in row]
                        nums = [n for n in nums if n and len(n) >= 6]
                        if len(nums) >= 2:
                            rows.append((nums[0], nums[1]))
                else:
                    # 如果没有表格，尝试提取文本
                    text = page.extract_text() or ""
                    for line in text.splitlines():
                        found = re.findall(r"(\d{6,})", line)
                        if len(found) >= 2:
                            rows.append((found[0], found[1]))
            except Exception as e:
                warnings.warn(f"处理第 {i} 页失败: {e}")

    return rows


def extract_from_pypdf2(pdf_path: str) -> list:
    """
    使用 PyPDF2 从 PDF 提取代码映射

    Args:
        pdf_path: PDF 文件路径

    Returns:
        代码映射列表 [(old_code, new_code), ...]
    """
    from PyPDF2 import PdfReader

    rows = []
    reader = PdfReader(pdf_path)
    pages = len(reader.pages)

    print(f"PDF 页数: {pages}")

    for i, page in enumerate(reader.pages, 1):
        print(f"处理第 {i}/{pages} 页...")
        try:
            text = page.extract_text() or ""
            for line in text.splitlines():
                found = re.findall(r"(\d{6,})", line)
                if len(found) >= 2:
                    rows.append((found[0], found[1]))
        except Exception as e:
            warnings.warn(f"处理第 {i} 页失败: {e}")

    return rows


def extract_code_mapping(pdf_path: str) -> list:
    """
    从 PDF 提取代码映射

    Args:
        pdf_path: PDF 文件路径

    Returns:
        代码映射列表 [(old_code, new_code), ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    print(f"开始提取代码映射: {pdf_path}")

    # 尝试使用 pdfplumber（推荐，支持表格提取）
    try:
        rows = extract_from_pdfplumber(pdf_path)
        print(f"使用 pdfplumber 提取到 {len(rows)} 条记录")
        return rows
    except ImportError:
        print("pdfplumber 未安装，尝试使用 PyPDF2...")
    except Exception as e:
        print(f"pdfplumber 提取失败: {e}，尝试使用 PyPDF2...")

    # 尝试使用 PyPDF2
    try:
        rows = extract_from_pypdf2(pdf_path)
        print(f"使用 PyPDF2 提取到 {len(rows)} 条记录")
        return rows
    except ImportError:
        raise ImportError("请安装 pdfplumber 或 PyPDF2 库: pip install pdfplumber")
    except Exception as e:
        raise RuntimeError(f"从 PDF 提取代码映射失败: {e}")


def save_code_mapping(rows: list, output_path: str):
    """
    保存代码映射到文本文件

    格式：
    旧代码    新代码

    Args:
        rows: 代码映射列表
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for old_code, new_code in rows:
            f.write(f"{old_code}\t{new_code}\n")

    print(f"代码映射已保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 50)
    print("北交所代码映射提取工具")
    print("=" * 50)

    # 检查 PDF 文件
    if not os.path.exists(PDF_PATH):
        print(f"\n错误: PDF 文件不存在")
        print(f"请将文件放置在: {PDF_PATH}")
        return

    print(f"\nPDF 文件: {PDF_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")

    # 提取代码映射
    print("\n开始提取...")
    rows = extract_code_mapping(PDF_PATH)

    if not rows:
        print("\n警告: 未提取到任何代码映射")
        return

    # 去重
    rows = list(set(rows))
    print(f"去重后: {len(rows)} 条记录")

    # 保存
    save_code_mapping(rows, OUTPUT_PATH)

    # 显示前10条记录
    print("\n前 10 条记录:")
    for i, (old_code, new_code) in enumerate(rows[:10], 1):
        print(f"{i:2d}. {old_code} -> {new_code}")

    if len(rows) > 10:
        print(f"... 还有 {len(rows) - 10} 条记录")

    print("\n" + "=" * 50)
    print("提取完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()