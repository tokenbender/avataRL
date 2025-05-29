#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>

struct Config {
    int64_t block_size = 256;
    int64_t vocab_size = 50257;
    int64_t n_layer = 6;
    int64_t n_head = 6;
    int64_t n_embed = 384;
};

struct TextLoader {
    TextLoader(int64_t B, int64_t T) : B(B), T(T), cur_pos(0) {
        std::ifstream in("/teamspace/studios/this_studio/input.txt");
        int64_t tok;
        while (in >> tok) tokens.push_back(tok);
        size_t N = tokens.size();
        std::cout << "loaded " << N << " tokens\n";
        std::cout << "processing " << (N / (B * T)) << " batches per epoch\n";
    }
    std::pair<torch::Tensor, torch::Tensor> next_batch() {
        int64_t span = B * T + 1;
        if (cur_pos + span > tokens.size()) cur_pos = 0;
        auto start = tokens.begin() + cur_pos;
        std::vector<int64_t> buf(start, start + span);
        cur_pos += B * T;
        auto data = torch::from_blob(buf.data(), {span}, torch::kInt64).clone();
        auto x = data.narrow(0, 0, B * T).view({B, T});
        auto y = data.narrow(0, 1, B * T).view({B, T});
        return {x, y};
    }
    int64_t B, T, cur_pos;
    std::vector<int64_t> tokens;
};

struct CausalSelfAttentionImpl : torch::nn::Module {
    CausalSelfAttentionImpl(const Config& cfg) {
        n_embed = cfg.n_embed;
        n_head = cfg.n_head;
        head_dim = n_embed / n_head;
        qkv = register_module("qkv", torch::nn::Linear(n_embed, 3 * n_embed));
        proj = register_module("proj", torch::nn::Linear(n_embed, n_embed));
        auto m = torch::ones({cfg.block_size, cfg.block_size}, torch::kBool).tril().view({1, 1, cfg.block_size, cfg.block_size});
        mask = m;
        register_buffer("mask", mask);
    }
    torch::Tensor forward(const torch::Tensor& x) {
        auto B = x.size(0), T = x.size(1);
        auto qkv_out = qkv->forward(x).view({B, T, 3, n_head, head_dim});
        auto q = qkv_out.select(2, 0).permute({0, 2, 1, 3});
        auto k = qkv_out.select(2, 1).permute({0, 2, 1, 3});
        auto v = qkv_out.select(2, 2).permute({0, 2, 1, 3});
        auto y = at::scaled_dot_product_attention(q, k, v, mask, 0.0, false);
        auto out = y.permute({0, 2, 1, 3}).contiguous().view({B, T, n_embed});
        return proj->forward(out);
    }
    int64_t n_embed, n_head, head_dim;
    torch::nn::Linear qkv{nullptr}, proj{nullptr};
    torch::Tensor mask;
};
TORCH_MODULE(CausalSelfAttention);

struct MLPImpl : torch::nn::Module {
    MLPImpl(const Config& cfg) {
        fc = register_module("fc", torch::nn::Linear(cfg.n_embed, 4 * cfg.n_embed));
        act = torch::nn::GELU();
        proj = register_module("proj", torch::nn::Linear(4 * cfg.n_embed, cfg.n_embed));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = fc->forward(x);
        x = act(x);
        return proj->forward(x);
    }
    torch::nn::Linear fc{nullptr}, proj{nullptr};
    torch::nn::GELU act;
};
TORCH_MODULE(MLP);

struct BlockImpl : torch::nn::Module {
    BlockImpl(const Config& cfg) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embed})));
        attn = register_module("attn", CausalSelfAttention(cfg));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embed})));
        mlp = register_module("mlp", MLP(cfg));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = x + attn->forward(ln1->forward(x));
        x = x + mlp->forward(ln2->forward(x));
        return x;
    }
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    CausalSelfAttention attn{nullptr};
    MLP mlp{nullptr};
};
TORCH_MODULE(Block);

struct GPTImpl : torch::nn::Module {
    GPTImpl(const Config& cfg) {
        wte = register_module("wte", torch::nn::Embedding(cfg.vocab_size, cfg.n_embed));
        wpe = register_module("wpe", torch::nn::Embedding(cfg.block_size, cfg.n_embed));
        for (int i = 0; i < cfg.n_layer; ++i) blocks->push_back(Block(cfg));
        register_module("blocks", blocks);
        ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embed})));
        lm_head = register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(cfg.n_embed, cfg.vocab_size).bias(false)));
        lm_head->weight = wte->weight;
        apply([](torch::nn::Module& m) {
            if (auto* L = m.as<torch::nn::Linear>())
                torch::nn::init::normal_(L->weight, 0.0, 0.02);
        });
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets) {
        auto B = idx.size(0), T = idx.size(1);
        auto pos = torch::arange(0, T, torch::kLong).to(idx.device());
        auto x = wte->forward(idx) + wpe->forward(pos);
        for (auto& m : *blocks) x = m->as<Block>()->forward(x);
        x = ln_f->forward(x);
        auto logits = lm_head->forward(x);
        torch::Tensor loss;
        if (targets.defined()) {
            loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(-1)}), targets.view(-1));
        }
        return {logits, loss};
    }
    torch::nn::Embedding wte{nullptr}, wpe{nullptr};
    torch::nn::ModuleList blocks;
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};
};
TORCH_MODULE(GPT);

int main() {
    Config cfg;
    int64_t B = 64, T = cfg.block_size;
    TextLoader loader(B, T);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    GPT model(cfg);
    model->to(device);
    torch::optim::AdamW opt(model->parameters(), 3e-4);
    for (int i = 0; i < 50; ++i) {
        auto [x, y] = loader.next_batch();
        x = x.to(device);
        y = y.to(device);
        opt.zero_grad();
        auto [logits, loss] = model->forward(x, y);
        loss.backward();
        opt.step();
        std::cout << "step " << i << " loss " << loss.item<double>() << '\n';
    }
    return 0;
}
