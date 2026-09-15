// main.cu
// أداة RIPEMD-160 مستقلة على GPU: تشغّل self-test إلزامي مقابل متجهات
// اختبار معروفة (KAT) قبل أي بحث فعلي، ثم تقيس الإنتاجية (Mhashes/s)
// وتقارن دفعات من مدخلات 32-بايت بالهدف hash160 المُعطى.
//
// ملاحظة مهمة: هذه الأداة تنفذ خطوة RIPEMD160(SHA256(pubkey)) الأخيرة
// فقط. لا تولّد مفاتيح عامة صالحة ضمن نطاق اللغز بنفسها — ذلك يتطلب
// طبقة EC (gen_startpubkeys.py + جمع نقاط تسلسلي) كما نوقش سابقًا.
// بدون تلك الطبقة، تشغيل هذه الأداة على بيانات عشوائية هو *قياس أداء
// فقط*، وليس بحثًا فعليًا عن اللغز.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>

extern "C" __global__ void ripemd160_batch_kernel(
    const uint8_t* inputs, uint8_t* outputs, const uint8_t* target20,
    int* match_count, long long* match_indices, long long n, int write_outputs);

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

static void hex_to_bytes(const char* hex, uint8_t* out, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        unsigned int b;
        sscanf(hex + i*2, "%2x", &b);
        out[i] = (uint8_t)b;
    }
}

static void bytes_to_hex(const uint8_t* b, int n, char* out) {
    for (int i = 0; i < n; i++) sprintf(out + i*2, "%02x", b[i]);
    out[n*2] = 0;
}

// إرسال إشعار Telegram عبر curl (مثبّت في صورة Docker). يقرأ التوكن
// ومعرّف المحادثة من متغيرات البيئة — لا شيء منها مكتوب داخل الكود.
// كل النصوص المُمرّرة هنا (hex أرقام فقط، من إنتاجنا الداخلي) لذا لا
// يوجد مدخل مستخدم غير موثوق يُمرَّر لـsystem().
static void telegram_notify(const char* text) {
    const char* token = getenv("TELEGRAM_BOT_TOKEN");
    const char* chat_id = getenv("TELEGRAM_CHAT_ID");
    if (!token || !chat_id || strlen(token) == 0 || strlen(chat_id) == 0) {
        fprintf(stderr, "[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID غير مضبوطة — تخطي الإشعار.\n");
        return;
    }
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "curl -s -m 10 -X POST \"https://api.telegram.org/bot%s/sendMessage\" "
        "-d chat_id=%s --data-urlencode text=\"%s\" > /dev/null 2>&1",
        token, chat_id, text);
    int rc = system(cmd);
    if (rc != 0) fprintf(stderr, "[telegram] تحذير: أمر curl أعاد كود خروج %d\n", rc);
}

// --- self-test: يجب أن يمر قبل أي تشغيل حقيقي ---
static void run_self_test() {
    // KAT1: 32 صفر بايت -> d1a70126ff7a149ca6f9b638db084480440ff842  (تحقق من الصحة أولاً: 20 بايت)
    uint8_t zero32[32]; memset(zero32, 0, 32);
    uint8_t ones32[32]; memset(ones32, 0xAA, 32);
    const char* expect1 = "d1a70126ff7a149ca6f9b638db084480440ff842";
    const char* expect2 = "cf7ff51392e9a37bc72c7284841db669c82e2c14";

    uint8_t h_inputs[64];
    memcpy(h_inputs, zero32, 32);
    memcpy(h_inputs + 32, ones32, 32);

    uint8_t *d_in, *d_out, *d_target;
    int *d_matchcount; long long *d_matchidx;
    CUDA_CHECK(cudaMalloc(&d_in, 64));
    CUDA_CHECK(cudaMalloc(&d_out, 40));
    CUDA_CHECK(cudaMalloc(&d_target, 20));
    CUDA_CHECK(cudaMalloc(&d_matchcount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matchidx, 1024*sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_matchcount, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_inputs, 64, cudaMemcpyHostToDevice));

    uint8_t dummy_target[20]; memset(dummy_target, 0, 20);
    CUDA_CHECK(cudaMemcpy(d_target, dummy_target, 20, cudaMemcpyHostToDevice));

    ripemd160_batch_kernel<<<1, 2>>>(d_in, d_out, d_target, d_matchcount, d_matchidx, 2, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t h_out[40];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 40, cudaMemcpyDeviceToHost));

    char hex1[41], hex2[41];
    bytes_to_hex(h_out, 20, hex1);
    bytes_to_hex(h_out + 20, 20, hex2);

    printf("[self-test] KAT1 got=%s expect=%s\n", hex1, expect1);
    printf("[self-test] KAT2 got=%s expect=%s\n", hex2, expect2);

    if (strcmp(hex1, expect1) != 0 || strcmp(hex2, expect2) != 0) {
        fprintf(stderr, "[self-test] FAILED — الكيرنل غير صحيح، لا يمكن الوثوق بأي نتيجة بحث. إيقاف.\n");
        telegram_notify("self-test FAILED على ripemd160-tool — الكيرنل غير صحيح، الجهاز متوقف. تحقق فورًا.");
        exit(2);
    }
    printf("[self-test] PASSED — منطق RIPEMD160 مطابق للمرجع المُختبر على CPU.\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_target);
    cudaFree(d_matchcount); cudaFree(d_matchidx);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <target_hash160_hex> [batch_size] [num_batches]\n", argv[0]);
        fprintf(stderr, "  مثال: %s f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 16777216 100\n", argv[0]);
        return 1;
    }
    if (strlen(argv[1]) != 40) {
        fprintf(stderr, "target hash160 يجب أن يكون 40 حرف hex بالضبط (20 بايت)\n");
        return 1;
    }

    run_self_test();

    uint8_t target[20];
    hex_to_bytes(argv[1], target, 20);

    long long batch = (argc > 2) ? atoll(argv[2]) : (1LL << 24); // 16.7M افتراضي
    long long num_batches = (argc > 3) ? atoll(argv[3]) : 100;

    uint8_t *d_in, *d_target;
    int *d_matchcount; long long *d_matchidx;
    CUDA_CHECK(cudaMalloc(&d_in, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_target, 20));
    CUDA_CHECK(cudaMalloc(&d_matchcount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matchidx, 1024 * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_target, target, 20, cudaMemcpyHostToDevice));

    uint8_t* h_in = (uint8_t*)malloc(batch * 32);
    long long* h_matchidx = (long long*)malloc(1024 * sizeof(long long));

    int threads = 256;
    long long blocks = (batch + threads - 1) / threads;

    printf("[بدء] target=%s batch=%lld batches=%lld threads/block=%d\n",
           argv[1], batch, num_batches, threads);
    printf("[تنبيه] المدخلات هنا عشوائية لأغراض القياس فقط — ليست مفاتيح عامة\n");
    printf("        صالحة ضمن نطاق اللغز. للبحث الفعلي، صِل هذه الأداة بمخرجات\n");
    printf("        مرحلة SHA256(pubkey) القادمة من التدرّج EC (-sp).\n\n");

    {
        char startup_msg[256];
        snprintf(startup_msg, sizeof(startup_msg),
            "ripemd160-tool: self-test PASSED، بدأ التشغيل. target=%s (وضع قياس أداء — بيانات عشوائية)",
            argv[1]);
        telegram_notify(startup_msg);
    }

    srand(12345);
    double total_time = 0;
    long long total_hashes = 0;

    for (long long b = 0; b < num_batches; b++) {
        for (long long i = 0; i < batch * 32; i++) h_in[i] = (uint8_t)(rand() & 0xFF);
        CUDA_CHECK(cudaMemcpy(d_in, h_in, batch * 32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_matchcount, 0, sizeof(int)));

        auto t0 = std::chrono::high_resolution_clock::now();
        ripemd160_batch_kernel<<<(int)blocks, threads>>>(
            d_in, nullptr, d_target, d_matchcount, d_matchidx, batch, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double>(t1 - t0).count();
        total_time += dt;
        total_hashes += batch;

        int h_matchcount = 0;
        CUDA_CHECK(cudaMemcpy(&h_matchcount, d_matchcount, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_matchcount > 0) {
            CUDA_CHECK(cudaMemcpy(h_matchidx, d_matchidx,
                       (h_matchcount < 1024 ? h_matchcount : 1024) * sizeof(long long),
                       cudaMemcpyDeviceToHost));
            printf("!!! تطابق في الدفعة %lld — عدد=%d\n", b, h_matchcount);

            // إرسال فوري قبل أي شيء آخر (نفس أولوية server.py الحالي:
            // لا تخاطر بفقدان النتيجة لو أُوقف instance vast.ai فجأة).
            char msg[256];
            snprintf(msg, sizeof(msg),
                "⚠️ تطابق hash160! batch=%lld index_in_batch=%lld target=%s "
                "(ملاحظة: بيانات هذه الدفعة عشوائية طالما لم يُربط الكيرنل بمولّد EC الفعلي — تحقق يدويًا)",
                b, h_matchidx[0], argv[1]);
            telegram_notify(msg);
        }

        double mhs = (batch / dt) / 1e6;
        printf("\rدفعة %lld/%lld | %.1f Mhashes/s | إجمالي %.2f G هاش", 
               b+1, num_batches, mhs, total_hashes / 1e9);
        fflush(stdout);
    }
    printf("\n\nمتوسط الإنتاجية: %.1f Mhashes/s\n", (total_hashes / total_time) / 1e6);

    free(h_in); free(h_matchidx);
    cudaFree(d_in); cudaFree(d_target); cudaFree(d_matchcount); cudaFree(d_matchidx);
    return 0;
}
