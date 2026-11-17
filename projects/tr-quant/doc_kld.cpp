// Belge düzeyinde KL ıraksaması: her belge kendi bağlamıyla baştan değerlendirilir ve bütün token'ları puanlanır,
// böylece Türkçe ve İngilizce aynı içerikte karşılaştırılır (llama-perplexity yalnız pencerenin ikinci yarısını
// puanlıyor, iki dilde puanlanan içerik farklı oluyordu).
// Kullanım: doc_kld TABAN.gguf TOKENLAR.bin ÇIKTI_ÖNEKİ N_THREAD VARYANT.gguf [VARYANT.gguf ...]
// TOKENLAR.bin: int32 belge sayısı, sonra her belge için int32 uzunluk ve int32 token kimlikleri (özel token'sız;
// model BOS istiyorsa burada eklenir). Çıktı: ÇIKTI_ÖNEKİ_<varyant>.csv, belge başına satır.
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<std::vector<llama_token>> read_docs(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "okunamadı: %s\n", path); std::exit(1); }
    int32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) std::exit(1);
    std::vector<std::vector<llama_token>> docs(n);
    for (auto & d : docs) {
        int32_t len = 0;
        if (std::fread(&len, 4, 1, f) != 1) std::exit(1);
        d.resize(len);
        if (std::fread(d.data(), 4, len, f) != (size_t) len) std::exit(1);
    }
    std::fclose(f);
    return docs;
}

struct Model {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    int n_vocab = 0;
    bool add_bos = false;
    llama_token bos = -1;

    Model(const char * path, int n_ctx, int n_threads) {
        auto mp = llama_model_default_params();
        // DOC_KLD_NGL: GPU'ya alınacak katman sayısı (CUDA derlemesinde 99); logit işleme her durumda CPU'da
        const char * ngl = std::getenv("DOC_KLD_NGL");
        mp.n_gpu_layers = ngl ? std::atoi(ngl) : 0;
        model = llama_model_load_from_file(path, mp);
        if (!model) { std::fprintf(stderr, "model yüklenemedi: %s\n", path); std::exit(1); }
        auto cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.n_ubatch = n_ctx;
        cp.n_seq_max = 1;
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { std::fprintf(stderr, "bağlam kurulamadı\n"); std::exit(1); }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        add_bos = llama_vocab_get_add_bos(vocab);
        bos = llama_vocab_bos(vocab);
        batch = llama_batch_init(n_ctx, 0, 1);
    }

    ~Model() {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
    }

    // Belgeyi değerlendirir; i. satırın logitleri seq[i+1]'i tahmin eder.
    std::vector<llama_token> eval(const std::vector<llama_token> & doc) {
        std::vector<llama_token> seq;
        if (add_bos) seq.push_back(bos);
        seq.insert(seq.end(), doc.begin(), doc.end());
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = (int32_t) seq.size();
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.token[i] = seq[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "llama_decode hatası\n"); std::exit(1); }
        return seq;
    }
};

// Taban modelin puanlanan her konumu için olasılıklar, negatif entropi, gerçek token'ın log-olasılığı, argmax.
struct BasePos {
    float negent;
    float lp_next;
    int32_t argmax;
};

static double logsumexp(const float * x, int n, float * mx_out) {
    float mx = *std::max_element(x, x + n);
    double s = 0.0;
    for (int v = 0; v < n; v++) s += std::exp((double) x[v] - mx);
    *mx_out = mx;
    return mx + std::log(s);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "kullanım: %s TABAN.gguf TOKENLAR.bin ÇIKTI_ÖNEKİ N_THREAD VARYANT.gguf [...]\n", argv[0]);
        return 1;
    }
    const auto docs = read_docs(argv[2]);
    const std::string prefix = argv[3];
    const int n_threads = std::atoi(argv[4]);
    size_t max_len = 0, n_pos = 0;
    for (const auto & d : docs) { max_len = std::max(max_len, d.size()); n_pos += d.size(); }
    const int n_ctx = (int) ((max_len + 1 + 255) / 256 * 256);
    ggml_backend_load_all();
    llama_backend_init();

    std::vector<float> probs;
    std::vector<BasePos> base;
    std::vector<size_t> offset(docs.size() + 1, 0);
    int n_vocab = 0;
    {
        Model m(argv[1], n_ctx, n_threads);
        n_vocab = m.n_vocab;
        probs.resize(n_pos * (size_t) n_vocab);
        base.resize(n_pos);
        size_t p = 0;
        for (size_t d = 0; d < docs.size(); d++) {
            offset[d] = p;
            const auto seq = m.eval(docs[d]);
            const int n = (int) seq.size() - 1;
#pragma omp parallel for num_threads(n_threads)
            for (int i = 0; i < n; i++) {
                const float * lg = llama_get_logits_ith(m.ctx, i);
                float mx;
                const double lse = logsumexp(lg, n_vocab, &mx);
                float * pr = probs.data() + (p + i) * (size_t) n_vocab;
                double negent = 0.0;
                int32_t am = 0;
                for (int v = 0; v < n_vocab; v++) {
                    const double lp = lg[v] - lse;
                    pr[v] = (float) std::exp(lp);
                    negent += pr[v] * lp;
                    if (lg[v] > lg[am]) am = v;
                }
                base[p + i] = {(float) negent, (float) (lg[seq[i + 1]] - lse), am};
            }
            p += n;
        }
        offset[docs.size()] = p;
        n_pos = p;
        std::fprintf(stderr, "taban: %zu belge, %zu puanlanan token, n_vocab %d\n", docs.size(), n_pos, n_vocab);
    }

    for (int a = 5; a < argc; a++) {
        std::string name = argv[a];
        name = name.substr(name.find_last_of('/') + 1);
        name = name.substr(0, name.rfind(".gguf"));
        Model m(argv[a], n_ctx, n_threads);
        if (m.n_vocab != n_vocab) { std::fprintf(stderr, "sözlük boyu farklı: %s\n", argv[a]); return 1; }
        FILE * out = std::fopen((prefix + "_" + name + ".csv").c_str(), "w");
        std::fprintf(out, "belge,token,kl_toplam,nll_taban,nll_varyant,ayni_top1\n");
        for (size_t d = 0; d < docs.size(); d++) {
            const auto seq = m.eval(docs[d]);
            const int n = (int) seq.size() - 1;
            if ((size_t) n != offset[d + 1] - offset[d]) { std::fprintf(stderr, "konum sayısı farklı\n"); return 1; }
            double kl = 0.0, nll_b = 0.0, nll_q = 0.0;
            int same = 0;
#pragma omp parallel for num_threads(n_threads) reduction(+ : kl, nll_b, nll_q, same)
            for (int i = 0; i < n; i++) {
                const size_t p = offset[d] + i;
                const float * lg = llama_get_logits_ith(m.ctx, i);
                float mx;
                const double lse = logsumexp(lg, n_vocab, &mx);
                const float * pr = probs.data() + p * (size_t) n_vocab;
                double cross = 0.0;
                int32_t am = 0;
                for (int v = 0; v < n_vocab; v++) {
                    cross += pr[v] * (double) lg[v];
                    if (lg[v] > lg[am]) am = v;
                }
                kl += base[p].negent - (cross - lse);
                nll_b -= base[p].lp_next;
                nll_q -= lg[seq[i + 1]] - lse;
                same += (am == base[p].argmax);
            }
            std::fprintf(out, "%zu,%d,%.9g,%.9g,%.9g,%d\n", d, n, kl, nll_b, nll_q, same);
        }
        std::fclose(out);
        std::fprintf(stderr, "bitti: %s\n", name.c_str());
    }
    llama_backend_free();
    return 0;
}
