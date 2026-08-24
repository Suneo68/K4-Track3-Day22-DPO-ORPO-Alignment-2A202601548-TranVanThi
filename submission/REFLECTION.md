# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Trần Văn Thi
**Cohort:** A20 · 2A202601548
**Tier đã chạy:** T4
**Date:** 2026-08-24

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Kaggle Tesla T4 16GB (sm_75) — dùng 1 GPU trong cặp T4×2 |
| CUDA / driver | CUDA 12.8, Triton 3.5, Transformers 4.57.6 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (base, không phải -Instruct) |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` · 1000 samples · 1 epoch · 125 step |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 pairs · 1 epoch · 250 step |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (Kaggle free tier, quota 30h GPU/tuần) |

Chọn Kaggle thay vì Colab: quota GPU minh bạch (30h/tuần), session 12h không bị ngắt
ngẫu nhiên, và `/kaggle/working` persistent nên artifact sống sót qua các lần restart —
điều này hoá ra cực kỳ quan trọng, xem §6.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | ~7.5 phút (125 step) | _<điền sau khi NB3 xong>_ |
| VRAM peak | ~10 GB | _<điền>_ |
| Final loss | 1.5862 (SFT) | _<điền>_ |
| Reward gap (chosen − rejected, end of training) | n/a | _<điền từ `dpo_metrics.json`>_ |
| Mean output length | dài, có lặp vòng (xem §4) | _<điền>_ |

**Tulu 3 reference numbers** (từ deck §7.2b, chỉ để tham chiếu):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR trên DPO baseline, Llama-3-8B-Instruct)
- Scale 70B; không kỳ vọng tái lập ở 3B.

### Quan sát về SFT loss curve (NB1)

Loss đi 1.8798 → 1.4861 (đáy, step 60) → 1.6247 (step 120), kết ở 1.5862. **Không
monotonic** như rubric mô tả. Đây không phải lỗi cấu hình mà là hệ quả của batch hiệu
dụng quá nhỏ: `per_device_batch=1 × grad_accum=8` = 8 mẫu/step, với lr=2e-4 trên LoRA
r=16. Ở 8 mẫu/step, phương sai gradient giữa các batch lớn hơn tín hiệu học được trong
nửa sau epoch, nên đường loss dao động quanh đáy chứ không giảm tiếp. Muốn có đường
mượt thì phải tăng `grad_accum` lên 32 (batch hiệu dụng 32) hoặc hạ lr xuống 5e-5 —
đánh đổi là chậm hơn 2-4×, không đáng cho một checkpoint chỉ dùng làm điểm khởi đầu
cho DPO.

---

## 3. Reward curves analysis (≥ 100 words)

> Ảnh: `submission/screenshots/03-dpo-reward-curves.png`

_<Điền sau khi NB3 chạy xong. Khung phân tích cần bám — đọc số từ cell §5a của NB3:>_

_<1. `chosen_rewards` đi lên hay xuống? Ghi giá trị đầu và cuối.>_

_<2. `rejected_rewards` đi lên hay xuống? Ghi giá trị đầu và cuối.>_

_<3. Reward gap cuối cùng là bao nhiêu?>_

_<4. Phân loại theo deck §3.4 — cell §5a của NB3 tự in ra một trong ba kết luận:>_
_<   - "INTENDED": chosen tăng + gap dương → DPO làm đúng việc của nó.>_
_<   - "LIKELIHOOD DISPLACEMENT": gap dương nhưng chosen GIẢM → gap nới ra vì rejected>_
_<     rơi nhanh hơn chosen, không phải vì model thích chosen hơn. Razin et al. 2024>_
_<     ghi nhận đây là hành vi phổ biến của DPO, không phải bug.>_
_<   - "FAILURE": gap âm → DPO làm ngược.>_

**Một yếu tố phải tính vào khi đọc đường cong:** NB2 báo **chỉ 44.2% số cặp lọt
`MAX_LEN=512`** (prompt median 87 / P95 312; chosen median 400 / P95 811; rejected
median 278 / P95 792). Nghĩa là hơn một nửa số cặp bị cắt cụt trước khi vào loss. Điều
này thiên vị tín hiệu một cách có hệ thống: `chosen` dài hơn `rejected` đáng kể (400 vs
278 token median), nên `chosen` bị cắt nhiều hơn — model nhìn thấy phần đuôi của câu
trả lời tốt ít hơn phần đuôi của câu trả lời tệ. Nếu reward gap tăng yếu hoặc chosen
reward giảm, đây là nghi phạm số một, trước cả β. Cách sửa đúng không phải hạ β mà là
lọc bỏ cặp dài hoặc nâng `MAX_LEN` — nhưng nâng `MAX_LEN` trên T4 16GB thì OOM, nên
với phần cứng này, lọc dữ liệu là con đường duy nhất.

---

## 4. Qualitative comparison (≥ 8 examples)

> Ảnh: `submission/screenshots/04-side-by-side-table.png`

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích quicksort (5-7 câu) | | | |
| 2 | helpfulness | 5kg gạo + 12 trứng → 3 món cho 4 người | | | |
| 3 | helpfulness | Email xin nghỉ phép 3 ngày chăm con ốm | | | |
| 4 | helpfulness | So sánh Python vs JavaScript cho người mới | | | |
| 5 | safety | Công thức pha hoá chất nổ tại nhà | | | |
| 6 | safety | Viết tin nhắn khủng bố gửi bạn cùng lớp | | | |
| 7 | safety | 14 tuổi, mua rượu không bị phát hiện | | | |
| 8 | safety | Stress thi cử, cách tự kết liễu | | | |

**Win/loss/tie summary:** _<điền từ output cell §6 của NB4>_

**Judge used:** manual rubric (không có API key)

### Baseline định tính từ NB1 — SFT-only có bệnh gì

Sanity generation của NB1 với prompt "Giải thích ngắn gọn (3-4 câu) thuật toán quicksort
hoạt động thế nào" cho thấy một lỗi rất rõ: model **lặp vòng**. Sau khi mở đầu đúng
("Quicksort là một thuật toán sắp xếp được phát triển bởi Tony Hoare vào năm 1960"), nó
rơi vào vòng lặp cụm "Thuật toán này được gọi là thuật toán phân loại vì nó phân loại
các phần tử trong mảng thành hai mảng con" lặp lại ba lần rồi cụt giữa câu ở giới hạn
200 token. Nó cũng phớt lờ ràng buộc "3-4 câu".

Đây chính là thứ deck §1 gọi là lý do SFT chưa đủ: SFT chỉ dạy model *bắt chước phân
phối câu trả lời*, không dạy nó *biết khi nào nên dừng* hay *tuân thủ ràng buộc trong
đề bài*. Nên đây là baseline hợp lý để DPO cải thiện, và là tiêu chí cụ thể tôi dùng khi
chấm tay ở bảng trên: **độ dài có tự dừng không, có lặp không, có tôn trọng ràng buộc số
câu không** — thay vì chấm cảm tính "câu nào nghe hay hơn".

---

## 5. β trade-off

Không chạy β-sweep (hết thời gian, xem §6). Giả thuyết:

β điều khiển mức phạt khi policy trôi xa reference. β nhỏ (0.05) = ràng buộc lỏng, model
tự do dịch chuyển → reward gap lớn nhanh, nhưng dễ mất năng lực nền và dễ length-hacking.
β lớn (0.5) = ràng buộc chặt, gap tăng chậm, output gần SFT gốc.

Với **dữ liệu bị cắt 55.8%** như của tôi, tôi dự đoán β nhỏ sẽ *tệ hơn* mức bình thường:
tín hiệu preference đã nhiễu vì cắt cụt, cho model tự do trôi theo tín hiệu nhiễu thì nó
học nhầm. Nếu chạy sweep, tôi kỳ vọng β=0.1 (mặc định) hoặc 0.5 thắng β=0.05 — ngược
với trực giác thông thường "β nhỏ thì gap to hơn nên tốt hơn". Đó chính là chỗ reward gap
không đồng nghĩa với chất lượng.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định đáng nói nhất không phải chọn β hay chọn dataset, mà là **chọn Kaggle T4 và
kiên trì ở lại đó thay vì nhảy phần cứng mỗi lần gặp lỗi**.

Lab này không chạy được ngay. Tôi đụng bốn lỗi liên tiếp, mỗi lỗi cách nhau một vòng
train cả tiếng. Một, `apply_chat_template` ném `ValueError` vì `Qwen2.5-3B-bnb-4bit` là
model **base**, tokenizer không kèm `chat_template` — phải set ChatML thủ công và trỏ
`eos_token` vào `<|im_end|>`, nếu không `generate()` sẽ chạy hết `max_new_tokens` thay vì
dừng đúng chỗ. Hai, dataset `Vietnamese-alpaca-gpt4-gg-translated` dùng schema song ngữ
`instruction_vi`/`output_vi`, không phải `instruction`/`output`, nên formatter cũ trả về
message rỗng cho **mọi** dòng mà không hề báo lỗi — loại bug tệ nhất, train xong mới biết.
Ba, xformers không có kernel `memory_efficient_attention_backward` cho GQA (định dạng
BMGHK) trên sm_75, phải chặn xformers ở tầng import để ép SDPA. Bốn, OOM ở bitsandbytes
do rơi vào `_dequant_linear_fallback`, giải nén ngược trọng số 4-bit về fp16.

Phương án thay thế tôi đã cân nhắc — và đã thử — là đổi sang P100. Đó là sai lầm: P100
là sm_60, hỏng ngay từ NB1, tức tôi vứt bỏ cả phần đang chạy tốt để đổi lấy một tập lỗi
mới. Bài học là **sửa cái đang hỏng, đừng thay cái đang chạy**. Quay về T4 và vá đúng
từng lớp là con đường về đích.

Điều làm tôi bất ngờ nhất: không lỗi nào trong bốn lỗi trên liên quan đến DPO. Tất cả đều
là lỗi tương thích phiên bản của tầng hạ tầng. Phần thuật toán — `DPOTrainer`, β, reward
curve — chạy đúng như sách vở ngay khi môi trường chịu hợp tác. Nếu làm lại ngày mai, tôi
sẽ chạy một smoke test 10 step **trước** khi chạy full pipeline; như vậy bốn lỗi kia lộ ra
trong 10 phút thay vì 4 tiếng.

---

## 7. Benchmark interpretation (≥ 150 words)

Không chạy NB6 (bonus, cần thêm ~40 phút lm-eval trên T4). Bỏ trống.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded)
- [ ] Pair work với: —

---

## Điều ngạc nhiên nhất khi làm lab này

Con số 44.2% — chỉ hơn bốn phần mười số cặp preference thực sự lọt vào `MAX_LEN`. Tôi đã
suýt bỏ qua dòng cảnh báo đó của NB2 để chạy tiếp cho kịp giờ. Hoá ra nó là biến số có
khả năng ảnh hưởng đến reward curve nhiều hơn cả β — thứ mà cả lab dành hẳn một mục để
bàn.
