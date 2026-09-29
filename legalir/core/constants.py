"""Hằng số và biểu thức chính quy dùng cho làm sạch/chunk văn bản pháp luật.

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

import re


EMBED_INSTRUCTION = (
    "Truy xuất văn bản pháp luật Việt Nam chứa đầy đủ căn cứ và quy định cần "
    "thiết để trả lời câu hỏi. Chú ý số điều, khoản, thời hạn, đối tượng và "
    "hiệu lực của quy định."
)


CHUNK_SCHEMA = 10


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


LEGAL_BOUNDARY_RE = re.compile(
    r"(?im)^(?=(?:(?:phần|chương|mục|tiểu mục|phụ lục)\s+[ivxlcdm\d]+\b|"
    r"điều\s+\d+[a-eđ]?(?:[.\s]|$)))"
)


HIERARCHY_HEADER_RE = re.compile(
    r"(?i)^(phần|chương|mục|tiểu mục)\s+[ivxlcdm\d]+\b"
)


HIERARCHY_CANONICAL_RE = re.compile(
    r"(?is)^(phần|chương|mục|tiểu mục)\s+([ivxlcdm\d]+)\s*[.:\-–—]?\s*(.*)$"
)


FRAGMENTED_HIERARCHY_LABEL_RE = re.compile(
    r"(?i)^(phần|chương|mục|tiểu mục)\s*$"
)


FRAGMENTED_HIERARCHY_ORDINAL_RE = re.compile(
    r"(?i)^([ivxlcdm]+|\d+)\s*[.:\-–—]?\s*$"
)


# Amendment laws insert articles such as Điều 6a/12b.  Treat the optional
# suffix as part of the legal boundary instead of dropping those articles
# into generic paragraph splitting.
ARTICLE_BOUNDARY_RE = re.compile(r"(?im)(?=^điều\s+\d+[a-eđ]?[.\s])")


ARTICLE_HEADER_RE = re.compile(r"(?im)^điều\s+\d+[a-eđ]?[.\s]")


# Khoản are the top-level numbered items directly under an Điều ("1.", "2."
# at line start); lettered sub-items (a), b)) belong to their parent Khoản
# and are intentionally not split further.
KHOAN_BOUNDARY_RE = re.compile(r"(?m)(?=^\d{1,2}\.\s)")


SPACE_RE = re.compile(r"[ \t\f\v]+")


BLANK_RE = re.compile(r"\n{3,}")


# Administrative tail matter is frequent in the supplied Vietnamese legal
# corpus and otherwise contaminates the final substantive article embedding.
NOI_NHAN_RE = re.compile(r"(?i)^nơi\s+nhận\s*:")


TITLE_SIGN_RE = re.compile(r"^(?:TM|KT|TL|Q)\.\s*[A-ZÀ-Ỹ]")


LUU_VT_RE = re.compile(r"(?i)^lưu\s*:\s*(?:vt|vp)\b")


DISTRIBUTION_LIST_RE = re.compile(r"^-\s")


BARE_SIGN_MARKER_RE = re.compile(r"^(?:TM|KT|TL|Q)\.?$")


GOV_TITLE_WHITELIST = {
    "THỦ TƯỚNG", "PHÓ THỦ TƯỚNG", "BỘ TRƯỞNG", "THỨ TRƯỞNG",
    "CHỦ TỊCH", "PHÓ CHỦ TỊCH", "THỐNG ĐỐC", "PHÓ THỐNG ĐỐC",
    "TỔNG GIÁM ĐỐC", "GIÁM ĐỐC", "PHÓ GIÁM ĐỐC", "CHÁNH ÁN", "PHÓ CHÁNH ÁN",
    "VIỆN TRƯỞNG", "CỤC TRƯỞNG", "VỤ TRƯỞNG", "TRƯỞNG BAN", "CHỦ NHIỆM",
    "PHÓ CHỦ NHIỆM",
}


INLINE_NOISE_RUN_RE = re.compile(r"[.\-_=~*#]{6,}")


# Preserve legal list structure instead of flattening numbered Khoản and
# lettered Điểm into the preceding sentence during OCR cleanup.
LEGAL_LIST_ITEM_RE = re.compile(r"(?i)^(?:\d{1,3}[.)]|[a-zđ][.)]|[-+•])\s+")


STOPWORDS = {
    "ai", "bao", "cho", "các", "có", "của", "được", "gì", "khi", "là",
    "một", "những", "nào", "phải", "quy", "định", "ra", "sao", "theo",
    "thì", "trong", "tại", "và", "về", "với", "việc", "như", "thế",
}


META_FEATURE_VERSION = "legalir_listwise_v2_top2_evidence"
