// Belebele'de GGUF türlerinin görev doğruluğu: belebele_fp8.py'nin düz metin yöntemiyle aynı puan, dört seçenek
// dizisinin ortak önekinden sonraki token'ların log-olasılık toplamı.
// Kullanım: belebele_gguf MODEL.gguf SORULAR.bin ÇIKTI.csv N_THREAD
// SORULAR.bin (belebele_gguf_prep.py): int32 soru sayısı; her soru için int32 ortak önek uzunluğu, int32 dizi
// sayısı (4), her dizi için int32 uzunluk ve int32 token kimlikleri (BOS dahil, tokenizer'ın ürettiği gibi).
// Çıktı: soru başına satır, dört puan.
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Item {
    int32_t common;
    std::vector<std::vector<llama_token>> seqs;
};

static std::vector<Item> read_items(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "okunamadı: %s\n", path); std::exit(1); }
    auto rd = [&](void * p, size_t n) { if (std::fread(p, 4, n, f) != n) std::exit(1); };
    int32_t n = 0;
    rd(&n, 1);
    std::vector<Item> items(n);
    for (auto & it : items) {
        int32_t k = 0;
        rd(&it.common, 1);
        rd(&k, 1);
        it.seqs.resize(k);
        for (auto & s : it.seqs) {
            int32_t len = 0;
            rd(&len, 1);
            s.resize(len);
            rd(s.data(), len);
        }
    }
    std::fclose(f);
    return items;
}

static double log_prob(const float * lg, int n_vocab, llama_token t) {
    const float mx = *std::max_element(lg, lg + n_vocab);
    double s = 0.0;
    for (int v = 0; v < n_vocab; v++) s += std::exp((double) lg[v] - mx);
    return lg[t] - (mx + std::log(s));
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "kullanım: %s MODEL.gguf SORULAR.bin ÇIKTI.csv N_THREAD\n", argv[0]);
        return 1;
    }
    const auto items = read_items(argv[2]);
    size_t max_len = 0;
    for (const auto & it : items)
        for (const auto & s : it.seqs) max_len = std::max(max_len, s.size());
    const int n_ctx = (int) ((max_len + 255) / 256 * 256);
    const int n_threads = std::atoi(argv[4]);

    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    const char * ngl = std::getenv("DOC_KLD_NGL");
    mp.n_gpu_layers = ngl ? std::atoi(ngl) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model yüklenemedi: %s\n", argv[1]); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "bağlam kurulamadı\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    // tokens[0..n) değerlendirilir; logits yalnız from'dan itibaren istenir
    auto eval = [&](const std::vector<llama_token> & tokens, int n, int from) {
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= from;
        }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "llama_decode hatası\n"); std::exit(1); }
    };

    FILE * out = std::fopen(argv[3], "w");
    std::fprintf(out, "soru,p0,p1,p2,p3\n");
    int n_single = 0;
    for (size_t q = 0; q < items.size(); q++) {
        const auto & it = items[q];
        const int c = it.common;
        bool single = true;
        for (const auto & s : it.seqs) single = single && (int) s.size() == c + 1;
        std::vector<double> score(it.seqs.size(), 0.0);
        if (single) {
            // dört dizi ortak önekten sonra tek token'da ayrışıyor: önek bir kez değerlendirilir
            n_single++;
            eval(it.seqs[0], c, c - 1);
            const float * lg = llama_get_logits_ith(ctx, c - 1);
            for (size_t k = 0; k < it.seqs.size(); k++) score[k] = log_prob(lg, n_vocab, it.seqs[k][c]);
        } else {
            for (size_t k = 0; k < it.seqs.size(); k++) {
                const auto & s = it.seqs[k];
                const int n = (int) s.size();
                eval(s, n, c - 1);
                for (int t = c; t < n; t++) score[k] += log_prob(llama_get_logits_ith(ctx, t - 1), n_vocab, s[t]);
            }
        }
        std::fprintf(out, "%zu", q);
        for (double v : score) std::fprintf(out, ",%.9g", v);
        std::fprintf(out, "\n");
    }
    std::fclose(out);
    std::fprintf(stderr, "bitti: %zu soru (%d tek token'da ayrışan)\n", items.size(), n_single);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
