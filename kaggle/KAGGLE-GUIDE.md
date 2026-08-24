# Chạy Lab 22 trên Kaggle — hướng dẫn từng bước

Notebook: [`Lab22_DPO_Kaggle_T4.ipynb`](Lab22_DPO_Kaggle_T4.ipynb) — bản T4 tier đã stitch sẵn
setup Kaggle + NB1→NB4 (core, 100 điểm) + cell đóng gói + NB5/NB6 (bonus).

**Vì sao Kaggle thay vì Colab:** T4×2 (dùng 1), 30h GPU/tuần có quota rõ ràng, session 12h
không bị disconnect ngẫu nhiên, `/kaggle/working` persistent giữa các lần chạy.

---

## 1. Chuẩn bị tài khoản (làm 1 lần)

Kaggle **bắt buộc verify số điện thoại** mới bật được Internet cho notebook. Không có Internet
thì không tải được model Qwen + dataset UltraFeedback → lab không chạy được.

`kaggle.com` → avatar → Settings → Phone Verification.

---

## 2. Tạo notebook

1. `kaggle.com/code` → **New Notebook**
2. File → **Import Notebook** → upload `kaggle/Lab22_DPO_Kaggle_T4.ipynb`
3. Panel bên phải (Notebook options), chỉnh đúng 3 mục:

| Mục | Đặt thành |
|---|---|
| Accelerator | **GPU T4 x2** (hoặc `GPU P100`) |
| Internet | **On** |
| Persistence | **Files only** |

> Nếu Accelerator vẫn để `None`, cell A3 sẽ assert fail ngay — đó là chủ ý, để bạn không
> phí 5 phút cài đặt rồi mới phát hiện thiếu GPU.

---

## 3. Chạy

**Cell A1 phải chạy đầu tiên, trước mọi thứ khác.** Nó set `CUDA_VISIBLE_DEVICES=0`.

Lý do: Kaggle cấp T4×2. Khi PyTorch thấy 2 GPU, HuggingFace `Trainer` tự bật DataParallel,
mà Unsloth không hỗ trợ multi-GPU → crash giữa lúc train NB1 hoặc NB3. Biến môi trường này
chỉ có tác dụng nếu được set **trước** khi `torch` được import lần đầu trong process.

Nếu lỡ import torch trước: `Run → Restart session` rồi chạy lại từ A1.

Trình tự và thời gian ước tính trên T4:

| Cell | Việc | Thời gian |
|---|---|---|
| A1–A5 | Setup + pip install | ~5 phút |
| NB1 | SFT-mini (1k VN Alpaca, 1 epoch) | ~10-15 phút |
| NB2 | Preference data prep | ~2 phút |
| NB3 | **DPO training** | ~15-25 phút |
| NB4 | Side-by-side 8 prompts × 2 model | ~5-8 phút |
| Z1 | Kiểm tra + zip artifact | ~10 giây |

Tổng core ~40-55 phút. Có thể **Run All** tới hết Z1 rồi đi làm việc khác.

Nếu cell A2 báo `Successfully installed torch-<version khác>`: `Run → Restart session`,
rồi chạy lại từ A1 (A2 lần 2 sẽ xong trong vài giây vì đã cache).

---

## 4. Lấy kết quả về

Cell **Z1** kiểm tra 9 artifact bắt buộc, in lại `dpo_metrics.json` (số để điền vào
REFLECTION §2 và §3), rồi zip thành `/kaggle/working/lab22-submission.zip`.

Tải về: panel phải → **Output** → `lab22-submission.zip` → Download.

Giải nén đè vào repo local. Cấu trúc trong zip khớp 1-1 với repo:

```
adapters/sft-mini/     adapters/dpo/     data/pref/     data/eval/
submission/screenshots/{02-sft-loss,03-dpo-reward-curves,04-side-by-side-table}.png
```

**Notebook đã chạy** (giữ output cells — rubric yêu cầu): File → Download Notebook,
lưu vào repo. Đặt tên gì cũng được, ví dụ `kaggle/Lab22_DPO_Kaggle_T4_executed.ipynb`.

---

## 5. Sau khi có artifact

Còn 3 việc không cần GPU:

1. Chụp thêm `01-setup-gpu.png` (output cell A1/A3 hiện tên GPU + VRAM) và
   `05-judge-output.png` (output judge hoặc rubric thủ công của NB4) vào `submission/screenshots/`.
2. Điền [`submission/REFLECTION.md`](../submission/REFLECTION.md) — 20/100 điểm nằm ở đây.
   §3 và §6 mỗi phần ≥150 từ. §3 phải nói về **cả** `chosen` lẫn `rejected` trajectory.
3. `python scripts/verify.py` → phải exit 0. Rồi commit + push repo public, paste URL vào LMS.

---

## 6. Sự cố hay gặp trên Kaggle

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `assert torch.cuda.device_count() == 1` fail ở A3 | torch đã được import trước A1. `Run → Restart session`, chạy lại từ A1. |
| `Unsloth: ... does not support multi GPU` | Y hệt trên — A1 chưa có tác dụng. |
| Không tải được model/dataset, lỗi network | Internet = Off, hoặc account chưa verify phone. |
| OOM ở NB3 step 1 | Sửa cell §0 của NB3: `MAX_LEN` 512→384, hoặc `GRAD_ACCUM` 8→16. Restart trước khi chạy lại. |
| `padding token is not set` | Đã xử lý sẵn trong notebook. Nếu vẫn gặp, bạn đã sửa nhầm cell tokenizer. |
| Session hết 12h / bị ngắt | Persistence = `Files only` giữ lại `/kaggle/working`. Chạy lại từ A1→A4 rồi tiếp từ notebook đang dở. |
| Hết quota GPU tuần (30h) | Quota reset thứ Bảy hàng tuần (giờ UTC). Core lab chỉ ~1h nên hiếm khi chạm trần. |
| `chosen_rewards` giảm mà gap tăng | **Không phải lỗi** — likelihood displacement (deck §3.4). Viết vào REFLECTION §3, đó là chỗ ăn điểm. |

---

## 7. Muốn nhanh hơn nữa?

Kaggle cũng cấp **P100 16GB** — nhanh hơn T4 khoảng 1.5-2× cho workload này vì băng thông
HBM2 cao hơn nhiều. Nếu hàng chờ T4 dài, đổi Accelerator sang `GPU P100`: notebook chạy y
nguyên, không cần sửa gì (P100 không hỗ trợ bf16 → code tự fallback fp16 qua
`torch.cuda.is_bf16_supported()`).

Tier BigGPU (Qwen2.5-7B) **không** chạy được trên Kaggle T4/P100 đơn — cần ~18GB VRAM.
Đừng đổi `COMPUTE_TIER` sang `BIGGPU` ở đây.
