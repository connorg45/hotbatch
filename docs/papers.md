# Primary References

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) introduces iteration-level scheduling and selective batching for generative-model serving.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://doi.org/10.1145/3600006.3613165) describes vLLM's block-based KV-cache management. Hotbatch uses a simpler fixed-slot cache and does not implement PagedAttention.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) studies scheduling interactions between prefill and decode work.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) describes the GPT-2 model family used by the server.
- [Candle](https://github.com/huggingface/candle) is the Rust tensor and neural-network framework used for GPT-2 execution.
