# Nghiên cứu: LegalIR – DSC 2026

Tổng hợp các thí nghiệm và kết luận đã đo được trong quá trình phát triển hệ thống.
Metric của cuộc thi là **Recall@5** (macro theo câu hỏi); 92% câu hỏi train (6.447/7.000) chỉ có
đúng 1 văn bản đúng (gold).

## 1. Bài toán và dữ liệu

| | |
|---|---|
| Kho văn bản | 8.532 văn bản pháp luật (luật, nghị định, thông tư, quyết định…) |
| Train | 7.000 câu hỏi có nhãn; giữ **1.000 câu làm validation** (tách ổn định theo hash) |
| Validation | 500 câu *development* (chọn checkpoint, trọng số) + 500 câu *locked* (chỉ báo cáo) |
| Public / private test | câu hỏi không nhãn; private có 2.080 câu |
| Đầu ra | đúng 5 CID mỗi câu hỏi |

Validation không tham gia train, mine hard negative hay chọn few-shot; câu train trùng nội dung
với validation được loại theo cả QID lẫn văn bản đã chuẩn hóa.

## 2. Quá trình phát triển

Điểm public theo các phiên bản chính:

| Phiên bản | Public |
|---|---:|
| ex9317 | 0.9317 |
| ex9377 | 0.9377 |
| ex9425 | 0.9425 |
| ex9453 | 0.9453 |
| ex9416 (R2 retrain 600 lỗi + 400 rehearsal) | 0.9416 |
| **ex9488** (B-repair, fusion cố định) | **0.9488** |
| Bài nộp cuối | 0.9498 public · 0.9495 private |

Các thay đổi lớn theo thời gian:

- **14/08** – Baseline BGE-M3 + LoRA A/B, FAISS, reranker BGE-reranker-v2-m3 LoRA (thay Qwen).
- **19/08** – Chunk theo Điều/Khoản thay cho gộp theo độ dài (mục 3); thêm dense C và reranker R2.
- **24/08** – rsLoRA rank 64 trên cả attention và FFN; evidence của mỗi văn bản lấy xen kẽ từ
  các retriever thay vì nối A→B→C rồi cắt.
- **29/08** – Khôi phục tên cho 1.125/8.532 văn bản có trường `name` rỗng từ URL; R1/R2 học trên
  tối đa 2 evidence/văn bản và gộp bằng max giống lúc suy luận.
- **30/08** – Chunk schema 10: sửa tiêu đề Chương bị OCR tách dòng; chọn checkpoint R1/R2 bằng
  Recall@5 thật trên pool cố định thay cho Hit@5 trên nhóm 1 positive.
- **13/09** – B-repair: C tiếp tục từ B, R2 tiếp tục từ R1 học mọi gold (multi-gold);
  fusion cố định B .4 / R1 .4 / R2 .2.

## 3. Chunking theo cấu trúc pháp lý

Đo trên 300 văn bản thật (seed cố định):

| Cách chunk | Số chunk | Chunk = đúng 1 Điều | Chunk trộn ≥ 2 Điều | Mảnh vỡ (không trọn Điều nào) |
|---|---:|---:|---:|---:|
| Gộp theo độ dài, 1.800 ký tự | 8.009 | 25,0% | 21,9% | 53,1% |
| Theo Điều/Khoản, 1.800 ký tự | 7.352 | 62,4% | 18,6% | 19,0% |
| **Theo Điều/Khoản, 1.000 ký tự** | 12.406 | **69,9%** | **7,6%** | 22,6% |

- Median độ dài một Điều chỉ 909 ký tự: gộp theo độ dài liên tục nhồi 2–3 Điều khác chủ đề vào một
  vector, làm loãng tín hiệu truy hồi.
- Chỉ đổi thuật toán (giữ ngưỡng 1.800) đã giảm mảnh vỡ từ 53% xuống 19%.
- Mỗi chunk mang lại đường dẫn *Phần/Chương/Mục* và tên văn bản; Chương/Mục là ranh giới cứng.
  Audit đủ 8.532 văn bản: lỗi tiền tố lặp giảm từ 72.696 chunk xuống 0; tổng 380.582 chunk.

## 4. Đóng góp của từng tín hiệu (validation 1.000 câu)

| Tín hiệu | Recall@5 |
|---|---:|
| Dense A | 0.9253 |
| Dense B | 0.9315 |
| Dense C | 0.8971 |
| Reranker R1 | 0.9469 |
| Reranker R2 | 0.9467 |
| Fusion B + R1 (không R2) | 0.9508 |
| **Fusion B + R1 + R2** | **0.9513** |

- Pool 60 văn bản (A/B/C hợp nhất) chứa gold với Recall@60 = **0.986**: phần lớn lỗi nằm ở khâu
  xếp hạng, không phải khâu truy hồi.
- R2 chỉ thêm +0.0005 so với fusion không R2; phần tăng nằm ở câu multi-gold.
- Câu multi-gold (7,9% số câu) chiếm **30%** tổng recall bị mất: Recall@5 multi-gold 0.813 so với
  single-gold 0.963.

## 5. Chọn trọng số fusion: đơn giản thắng

| Bộ trọng số | Validation | Public |
|---|---:|---:|
| **fixed** B .4, R1 .4, R2 .2 | 0.9513 | **0.9488** |
| best_val (simplex + tinh chỉnh) | 0.9544 | 0.9443 |
| positive_grid (100.000 bộ) | 0.9544 | 0.9443 |

Tìm trọng số trên 500 câu development tăng validation +0.003 nhưng **giảm public −0.0045**; hai bộ
tối ưu đổi top-5 của ~33% câu so với fixed. Kết luận: với tập development nhỏ, lưới trọng số dày
khai thác nhiễu; bộ trọng số ít tham số, cố định, tổng quát hóa tốt hơn. Bài nộp dùng bộ fixed.

## 6. Phân rã lỗi còn lại

Trên 1.000 câu validation còn 69 cặp (câu, gold) bị trượt; vị trí của gold trong ranking cuối:

| Gold nằm ở | Số cặp | Recall tối đa cứu được |
|---|---:|---:|
| hạng 6–10 | 22 (32%) | +0.0172 |
| hạng 11–20 | 17 (25%) | +0.0127 |
| hạng 21+ | 9 (13%) | +0.0051 |
| **ngoài pool** | **21 (30%)** | +0.0132 |

- 30% lỗi nằm ngoài pool: không thao tác rerank nào cứu được, chỉ cải thiện retrieval.
- Vì metric là Recall@5, đổi thứ tự *bên trong* top-5 không đổi điểm; chỉ đưa gold từ hạng ≥ 6 vào
  top-5 mới tăng điểm, mỗi câu private cứu được đáng +1/2080 = +0.00048.

## 7. Những hướng đã thử và không hiệu quả

- **Chuyển nhãn từ câu train gần giống** (IDF-Jaccard trên câu hỏi): ngay cả câu trùng khít, gold
  của câu train chỉ đúng 83% (nhãn train có ~17% nhiễu). Khi văn bản chuyển sang *không* có sẵn trong
  top-5, xác suất đúng chỉ 4–45% tùy ngưỡng; đọc tay 16 ca tương đồng cao đều sai (cùng khuôn câu
  hỏi nhưng khác chủ thể/ngành, hoặc nhãn rác).
- **Tín hiệu lexical (BM25, BLEU) làm tầng hai** cho hạng 6–10: khi bất đồng với top-5 của fusion,
  tín hiệu độc lập gần như luôn sai vì hệ thống đã ở recall 0.95.
- **Dense C huấn luyện lại**: không tăng key chọn checkpoint, pool được chọn đặt C = 0.
- **R2 với lịch 600 lỗi + 400 rehearsal**: làm R2 lệch, public giảm từ 0.9453 xuống 0.9416.

## 8. Hướng phát triển

1. **Retrieval recall**: 30% lỗi nằm ngoài pool. Thêm ứng viên từ BM25 / sparse head của BGE-M3
   (giữ nguyên 60 CID cũ, chỉ bổ sung) và đo số gold mới vào pool.
2. **Evidence tốt hơn cho reranker**: thêm chunk BM25 *bên trong* mỗi văn bản ứng viên bên cạnh 4
   chunk dense, giữ nguyên trọng số để đo riêng tác động.
3. **Multi-gold**: chiếm 30% loss dù chỉ 8% số câu; cần mục tiêu huấn luyện và cách lấy top-5 đa
   dạng văn bản hơn (ví dụ luật gốc + văn bản hướng dẫn).

## Tài liệu tham khảo

- Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
  Embeddings*, 2024. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank
  learning methods*, SIGIR 2009.
- Li et al., *PARADE: Passage Representation Aggregation for Document Reranking*, 2020.
  [arXiv:2008.09093](https://arxiv.org/abs/2008.09093)
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models*, 2021.
  [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
