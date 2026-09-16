// ripemd160.cu
// كيرنل CUDA لحساب RIPEMD-160 لمدخل ثابت الطول (32 بايت = ناتج SHA256).
// منطق الدوال هنا منقول سطرًا بسطر من نسخة C تم التحقق من صحتها
// مقابل متجهات اختبار رسمية ("", "abc", "message digest") + متجهات
// KAT مُولّدة محليًا لمدخل 32 بايت (انظر ripemd160_ref.c / kat.c).
// هذا التطابق الحرفي بين النسختين هو ما يقلل خطر وجود عيب في الرياضيات
// عند النقل لـCUDA، لكنه لا يغني عن self-test عند الإقلاع (main.cu).

#include <cstdint>

#define ROL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

__device__ __forceinline__ uint32_t F(int j, uint32_t x, uint32_t y, uint32_t z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

__constant__ uint32_t d_KL[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
__constant__ uint32_t d_KR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

__constant__ int d_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__constant__ int d_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__constant__ int d_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__constant__ int d_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

// يحسب RIPEMD160 لمدخل 32 بايت بالضبط (كتلة واحدة بعد الحشو الثابت).
// in: مؤشر لـ32 بايت. out: مؤشر لـ20 بايت ناتج.
__device__ void ripemd160_fixed32(const uint8_t *in, uint8_t *out) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X[i] = (uint32_t)in[i*4] | ((uint32_t)in[i*4+1] << 8) |
               ((uint32_t)in[i*4+2] << 16) | ((uint32_t)in[i*4+3] << 24);
    }
    // حشو ثابت لمدخل 32 بايت: 0x80 ثم أصفار ثم طول الرسالة بالبت (256) كـ 64-بت LE
    X[8]  = 0x00000080u;
    X[9]  = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 0x00000100u; // 256 بت
    X[15] = 0;

    uint32_t al=h0, bl=h1, cl=h2, dl=h3, el=h4;
    uint32_t ar=h0, br=h1, cr=h2, dr=h3, er=h4;

    #pragma unroll
    for (int j = 0; j < 80; j++) {
        uint32_t t = ROL(al + F(j, bl, cl, dl) + X[d_RL[j]] + d_KL[j/16], d_SL[j]) + el;
        al = el; el = dl; dl = ROL(cl, 10); cl = bl; bl = t;
        t = ROL(ar + F(79 - j, br, cr, dr) + X[d_RR[j]] + d_KR[j/16], d_SR[j]) + er;
        ar = er; er = dr; dr = ROL(cr, 10); cr = br; br = t;
    }

    uint32_t t = h1 + cl + dr;
    h1 = h2 + dl + er; h2 = h3 + el + ar; h3 = h4 + al + br; h4 = h0 + bl + cr; h0 = t;

    uint32_t H[5] = {h0, h1, h2, h3, h4};
    #pragma unroll
    for (int i = 0; i < 5; i++)
        #pragma unroll
        for (int b = 0; b < 4; b++)
            out[i*4+b] = (uint8_t)(H[i] >> (8*b));
}

// كيرنل يعالج دفعة كاملة: كل thread ياخذ مدخل 32 بايت، يحسب الهاش،
// ويقارنه بالهدف الثابت (d_target). عند تطابق يسجل الفهرس في d_matches.
extern "C" __global__ void ripemd160_batch_kernel(
    const uint8_t* __restrict__ inputs,   // n * 32 بايت
    uint8_t* __restrict__ outputs,        // n * 20 بايت (لأغراض self-test فقط)
    const uint8_t* __restrict__ target20, // 20 بايت هدف ثابت
    int* __restrict__ match_count,
    long long* __restrict__ match_indices,
    long long n,
    int write_outputs)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint8_t out[20];
    ripemd160_fixed32(inputs + idx*32, out);

    if (write_outputs) {
        #pragma unroll
        for (int i = 0; i < 20; i++) outputs[idx*20+i] = out[i];
    }

    bool match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (out[i] != target20[i]) { match = false; break; }
    }
    if (match) {
        int slot = atomicAdd(match_count, 1);
        if (slot < 1024) match_indices[slot] = idx; // سقف أمان 1024 تطابق لكل دفعة
    }
}

// --- مسار قياس الأداء الخالص على GPU (بدون أي بيانات من CPU) ---
// كل thread يولّد مدخله (32 بايت) بنفسه محليًا عبر splitmix64 (PRNG
// سريع وبسيط، غير تشفيري — لا حاجة لجودة تشفيرية هنا، فقط بيانات
// "تبدو عشوائية" لقياس سرعة RIPEMD160 الفعلية على GPU بلا أي عنق
// اختناق من CPU أو من نقل بيانات عبر PCIe).
__device__ __forceinline__ uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void ripemd160_random_batch_kernel(
    unsigned long long seed_base,         // يتغيّر كل دفعة (مثلاً رقم الدفعة) لتفادي تكرار نفس المدخلات
    const uint8_t* __restrict__ target20,
    int* __restrict__ match_count,
    long long* __restrict__ match_indices,
    long long n)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // بذرة فريدة لكل thread: مزيج seed_base مع idx
    uint64_t state = seed_base ^ ((uint64_t)idx * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);

    uint8_t in[32];
    #pragma unroll
    for (int w = 0; w < 4; w++) {
        uint64_t r = splitmix64(state);
        #pragma unroll
        for (int b = 0; b < 8; b++) in[w*8+b] = (uint8_t)(r >> (8*b));
    }

    uint8_t out[20];
    ripemd160_fixed32(in, out);

    bool match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (out[i] != target20[i]) { match = false; break; }
    }
    if (match) {
        int slot = atomicAdd(match_count, 1);
        if (slot < 1024) match_indices[slot] = idx;
    }
}
