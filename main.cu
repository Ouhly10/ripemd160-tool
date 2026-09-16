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
#include <ctime>
#include <string>

extern "C" __global__ void ripemd160_batch_kernel(
    const uint8_t* inputs, uint8_t* outputs, const uint8_t* target20,
    int* match_count, long long* match_indices, long long n, int write_outputs);

extern "C" __global__ void ripemd160_random_batch_kernel(
    unsigned long long seed_base, const uint8_t* target20,
    int* match_count, long long* match_indices, long long n);

static void telegram_notify(const char* text); // تعريف أمامي — CUDA_CHECK يستخدمه قبل تعريفه الكامل أدناه

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    char _msg[256]; snprintf(_msg, sizeof(_msg), \
        "خطأ CUDA في ripemd160-tool (%s:%d): %s — الجهاز/التعريف قد يكون غير متوافق. توقفت الحاوية.", \
        __FILE__, __LINE__, cudaGetErrorString(e)); \
    telegram_notify(_msg); \
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
        fprintf(stderr, "  num_batches=0 أو غير مُمرَّر يعني: تشغيل مستمر بلا توقف (الوضع الافتراضي)\n");
        fprintf(stderr, "  مثال (لا نهائي):  %s f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 16777216\n", argv[0]);
        fprintf(stderr, "  مثال (محدود):     %s f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 16777216 100\n", argv[0]);
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
    // num_batches <= 0 يعني تشغيل لا نهائي — هذا الوضع الافتراضي الآن،
    // لاستغلال GPU بنسبة قريبة من 100% طوال مدة الإيجار الفعلية بدل
    // التوقف والانتظار (sleep) في loop.sh بين كل دورة قصيرة.
    long long num_batches = (argc > 3) ? atoll(argv[3]) : 0;
    bool infinite = (num_batches <= 0);

    uint8_t *d_target;
    int *d_matchcount; long long *d_matchidx;
    CUDA_CHECK(cudaMalloc(&d_target, 20));
    CUDA_CHECK(cudaMalloc(&d_matchcount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matchidx, 1024 * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_target, target, 20, cudaMemcpyHostToDevice));

    long long* h_matchidx = (long long*)malloc(1024 * sizeof(long long));

    int threads = 256;
    long long blocks = (batch + threads - 1) / threads;

    printf("[بدء] target=%s batch=%lld batches=%s threads/block=%d\n",
           argv[1], batch, infinite ? "لا نهائي" : std::to_string(num_batches).c_str(), threads);
    printf("[تنبيه] المدخلات هنا عشوائية (مولّدة بالكامل على GPU عبر splitmix64)\n");
    printf("        لأغراض القياس فقط — ليست مفاتيح عامة صالحة ضمن نطاق اللغز.\n");
    printf("        للبحث الفعلي، صِل هذه الأداة بمخرجات مرحلة SHA256(pubkey)\n");
    printf("        القادمة من التدرّج EC (-sp).\n\n");

    {
        // أرسل رسالة "بدء التشغيل" مرة واحدة فقط طوال عمر هذه الحاوية —
        // مهم بشكل خاص الآن أن الحلقة لا نهائية أصلاً بشكل طبيعي، لكن
        // نُبقي الفحص كشبكة أمان لو أعاد loop.sh تشغيل العملية بعد انهيار.
        const char* marker_path = "/tmp/.ripemd160_notified_startup";
        FILE* marker = fopen(marker_path, "r");
        if (!marker) {
            char startup_msg[256];
            snprintf(startup_msg, sizeof(startup_msg),
                "ripemd160-tool: self-test PASSED، بدأ التشغيل المستمر (توليد عشوائي على GPU بالكامل). target=%s",
                argv[1]);
            telegram_notify(startup_msg);
            FILE* mk = fopen(marker_path, "w");
            if (mk) fclose(mk);
        } else {
            fclose(marker);
        }
    }

    double total_time = 0;
    long long total_hashes = 0;
    double window_time = 0;
    long long window_hashes = 0;

    for (long long b = 0; infinite || b < num_batches; b++) {
        CUDA_CHECK(cudaMemset(d_matchcount, 0, sizeof(int)));

        // seed_base يتغيّر كل دفعة (وقت التشغيل + رقم الدفعة) لتفادي
        // تكرار نفس المدخلات عبر دورات متعددة
        unsigned long long seed_base = (unsigned long long)time(nullptr) * 1000003ULL + (unsigned long long)b;

        auto t0 = std::chrono::high_resolution_clock::now();
        ripemd160_random_batch_kernel<<<(int)blocks, threads>>>(
            seed_base, d_target, d_matchcount, d_matchidx, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double>(t1 - t0).count();
        total_time += dt;
        total_hashes += batch;
        window_time += dt;
        window_hashes += batch;

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
        printf("\rدفعة %lld%s | %.1f Mhashes/s | إجمالي %.2f G هاش",
               b+1, infinite ? "" : (std::string("/") + std::to_string(num_batches)).c_str(),
               mhs, total_hashes / 1e9);
        fflush(stdout);

        // تقرير دوري كل 1000 دفعة بسطر منفصل (وليس \r) — مفيد خاصة في
        // الوضع اللانهائي لمتابعة التقدم في سجلات vast.ai بمرور الوقت.
        if ((b + 1) % 1000 == 0) {
            printf("\n[تقرير] بعد %lld دفعة: متوسط آخر 1000 دفعة = %.1f Mhashes/s | إجمالي منذ البدء = %.2f G هاش\n",
                   b + 1, (window_hashes / window_time) / 1e6, total_hashes / 1e9);
            fflush(stdout);
            window_time = 0;
            window_hashes = 0;
        }
    }
    printf("\n\nمتوسط الإنتاجية: %.1f Mhashes/s\n", (total_hashes / total_time) / 1e6);

    free(h_matchidx);
    cudaFree(d_target); cudaFree(d_matchcount); cudaFree(d_matchidx);
    return 0;
}
