# VLLM Architecture Overview

## Project Description

VLLM is a high-throughput and memory-efficient inference engine for Large Language Models (LLMs). It provides easy, fast, and cheap LLM serving for everyone with state-of-the-art features like PagedAttention, continuous batching, and distributed execution.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        CLI[CLI Interface]
        API[OpenAI API Server]
        LLM[LLM Class]
        SDK[Python SDK]
    end
    
    subgraph "Engine Layer"
        LE[LLM Engine]
        ALE[Async LLM Engine]
        MPE[Multiprocessing Engine]
    end
    
    subgraph "Core Components"
        SCH[Scheduler]
        EXE[Executor]
        WRK[Workers]
        IPR[Input Processor]
        OPR[Output Processor]
    end
    
    subgraph "Model Execution"
        ME[Model Executor]
        ML[Model Loader]
        ATT[Attention Layer]
        QUA[Quantization]
        MOD[Model Registry]
    end
    
    subgraph "Memory Management"
        CM[Cache Manager]
        PA[PagedAttention]
        BSM[Block Space Manager]
        KVC[KV Cache]
    end
    
    subgraph "Distributed Computing"
        DC[Device Communicators]
        PS[Parallel State]
        KVT[KV Transfer]
        RAY[Ray Integration]
    end
    
    subgraph "Specialized Features"
        SD[Spec Decode]
        LORA[LoRA Support]
        MM[Multimodal]
        PROF[Profiler]
    end
    
    CLI --> LE
    API --> ALE
    LLM --> LE
    SDK --> LE
    
    LE --> SCH
    ALE --> SCH
    MPE --> SCH
    
    SCH --> EXE
    EXE --> WRK
    SCH --> IPR
    SCH --> OPR
    
    WRK --> ME
    ME --> ML
    ME --> ATT
    ME --> QUA
    ML --> MOD
    
    ATT --> PA
    PA --> CM
    CM --> BSM
    BSM --> KVC
    
    EXE --> DC
    WRK --> PS
    DC --> KVT
    EXE --> RAY
    
    ME --> SD
    ME --> LORA
    ME --> MM
    WRK --> PROF
```

## Component Details

### 1. Entry Points (`vllm/entrypoints/`)

The entry points provide different interfaces for accessing VLLM functionality:

```mermaid
graph LR
    subgraph "Entry Points"
        API[api_server.py<br/>OpenAI Compatible API]
        LLM[llm.py<br/>Simple Python Interface]
        CLI[cli/<br/>Command Line Tools]
        CHAT[chat_utils.py<br/>Chat Utilities]
        SCORE[score_utils.py<br/>Scoring Utils]
    end
    
    subgraph "OpenAI Serving"
        OCHAT[serving_chat.py]
        OCOMP[serving_completion.py]
        OEMB[serving_embedding.py]
        OTOK[serving_tokenization.py]
        OMOD[serving_models.py]
    end
    
    API --> OCHAT
    API --> OCOMP
    API --> OEMB
    API --> OTOK
    API --> OMOD
    
    LLM --> ENGINE[LLM Engine]
    CLI --> ENGINE
    API --> ENGINE
```

**Key Functions:**
- **api_server.py**: FastAPI-based OpenAI-compatible API server
- **llm.py**: Simple Python class interface for direct model usage
- **cli/**: Command-line interfaces for serving and inference
- **OpenAI serving modules**: Handle different OpenAI API endpoints (chat, completion, embedding, etc.)

### 2. Engine Layer (`vllm/engine/`)

The engine layer manages the overall execution flow and request handling:

```mermaid
graph TB
    subgraph "Engine Components"
        LE[llm_engine.py<br/>Synchronous Engine]
        ALE[async_llm_engine.py<br/>Asynchronous Engine]
        MPE[multiprocessing/<br/>Multiprocess Engine]
        ARGS[arg_utils.py<br/>Configuration]
        PROT[protocol.py<br/>Interfaces]
    end
    
    subgraph "Output Processing"
        OPI[output_processor/<br/>interfaces.py]
        OPS[output_processor/<br/>stop_checker.py]
        OPU[output_processor/<br/>util.py]
    end
    
    LE --> SCHEDULER[Core Scheduler]
    ALE --> SCHEDULER
    MPE --> SCHEDULER
    
    LE --> OPI
    ALE --> OPI
    
    ARGS --> LE
    ARGS --> ALE
    PROT --> LE
```

**Key Functions:**
- **LLMEngine**: Core synchronous engine handling request processing
- **AsyncLLMEngine**: Asynchronous wrapper for concurrent request handling
- **Multiprocessing Engine**: Enables multi-process execution for better scalability
- **Output Processors**: Handle response formatting, stopping criteria, and result aggregation

### 3. Core Scheduler (`vllm/core/`)

The scheduler manages request batching, memory allocation, and execution ordering:

```mermaid
graph TB
    subgraph "Scheduler Components"
        SCH[scheduler.py<br/>Main Scheduler]
        INT[interfaces.py<br/>Block Manager Interface]
        BSM[block_space_manager.py<br/>Memory Block Management]
    end
    
    subgraph "Scheduling Features"
        BATCH[Continuous Batching]
        PREEMPT[Preemption<br/>Swap/Recompute]
        BUDGET[Resource Budgeting]
        CACHE[Prefix Caching]
    end
    
    SCH --> BATCH
    SCH --> PREEMPT
    SCH --> BUDGET
    SCH --> CACHE
    
    SCH --> BSM
    BSM --> INT
```

**Key Functions:**
- **Request Scheduling**: Intelligent batching of requests for optimal throughput
- **Memory Management**: Efficient allocation and deallocation of GPU memory blocks
- **Preemption**: Swapping or recomputing sequences when memory is needed
- **Prefix Caching**: Caching common prefixes to avoid recomputation

### 4. Execution Layer (`vllm/executor/` & `vllm/worker/`)

Manages distributed execution across multiple devices:

```mermaid
graph TB
    subgraph "Executors"
        UNI[uniproc_executor.py<br/>Single Process]
        RAY[ray_distributed_executor.py<br/>Ray Distribution]
        MP[mp_distributed_executor.py<br/>Multiprocess Distribution]
        BASE[executor_base.py<br/>Base Interface]
    end
    
    subgraph "Workers"
        WB[worker_base.py<br/>Base Worker]
        WRK[worker.py<br/>GPU Worker]
        CPU[cpu_worker.py<br/>CPU Worker]
        TPU[tpu_worker.py<br/>TPU Worker]
        HPU[hpu_worker.py<br/>HPU Worker]
        XPU[xpu_worker.py<br/>XPU Worker]
        NEU[neuron_worker.py<br/>Neuron Worker]
    end
    
    subgraph "Specialized Workers"
        MS[multi_step_worker.py<br/>Multi-step Processing]
        SPEC[Speculative Decoding Workers]
    end
    
    UNI --> WRK
    RAY --> WRK
    MP --> WRK
    
    WB --> WRK
    WB --> CPU
    WB --> TPU
    WB --> HPU
    WB --> XPU
    WB --> NEU
    
    WRK --> MS
    MS --> SPEC
```

**Key Functions:**
- **Distributed Execution**: Support for tensor parallelism and pipeline parallelism
- **Multi-Device Support**: GPU, CPU, TPU, HPU, XPU, and Neuron backends
- **Ray Integration**: Scalable distributed computing with Ray
- **Specialized Workers**: Multi-step processing and speculative decoding

### 5. Model Executor (`vllm/model_executor/`)

Handles model loading, execution, and optimization:

```mermaid
graph TB
    subgraph "Model Management"
        REG[models/registry.py<br/>Model Registry]
        LOAD[model_loader.py<br/>Model Loader]
        INT[models/interfaces.py<br/>Model Interfaces]
    end
    
    subgraph "Model Layers"
        LAY[layers/<br/>Neural Network Layers]
        SAMP[layers/sampler.py<br/>Sampling Logic]
        QUANT[layers/quantization/<br/>Quantization Support]
        ATT[Attention Layers]
    end
    
    subgraph "Optimizations"
        GD[guided_decoding/<br/>Guided Generation]
        COP[custom_op.py<br/>Custom Operations]
        UTIL[utils.py<br/>Utilities]
    end
    
    subgraph "Supported Models"
        LLAMA[LLaMA]
        MISTRAL[Mistral]
        QWEN[Qwen]
        GEMMA[Gemma]
        OTHER[Many Others...]
    end
    
    REG --> LLAMA
    REG --> MISTRAL
    REG --> QWEN
    REG --> GEMMA
    REG --> OTHER
    
    LOAD --> REG
    LOAD --> LAY
    LAY --> SAMP
    LAY --> QUANT
    LAY --> ATT
    
    LOAD --> GD
    LOAD --> COP
```

**Key Functions:**
- **Model Registry**: Central registry of supported model architectures
- **Dynamic Loading**: Runtime model loading with configuration validation
- **Quantization**: Support for GPTQ, AWQ, AutoRound, INT4/INT8/FP8
- **Custom Kernels**: Optimized CUDA kernels for better performance

### 6. Attention & Memory Management (`vllm/attention/`)

Core memory-efficient attention implementation:

```mermaid
graph TB
    subgraph "Attention Backends"
        FLASH[flash_attn.py<br/>FlashAttention]
        FLASHINF[flashinfer.py<br/>FlashInfer]
        TORCH[torch_sdpa.py<br/>PyTorch SDPA]
        XFORM[xformers.py<br/>xFormers]
        CPU[CPU Attention]
        BLOCK[blocksparse_attn.py<br/>Block Sparse]
    end
    
    subgraph "Memory Management"
        PA[PagedAttention<br/>Core Innovation]
        CACHE[KV Cache<br/>Management]
        BLOCK_MGR[Block Allocation]
        SWAP[CPU Swapping]
    end
    
    subgraph "Optimization Features"
        CONT[Continuous Batching]
        PREFIX[Prefix Caching]
        CHUNK[Chunked Prefill]
        GRAPH[CUDA Graphs]
    end
    
    PA --> FLASH
    PA --> FLASHINF
    PA --> TORCH
    PA --> XFORM
    
    PA --> CACHE
    CACHE --> BLOCK_MGR
    BLOCK_MGR --> SWAP
    
    PA --> CONT
    PA --> PREFIX
    PA --> CHUNK
    PA --> GRAPH
```

**Key Functions:**
- **PagedAttention**: Revolutionary memory management for attention computation
- **Multiple Backends**: Support for various attention implementations
- **Continuous Batching**: Dynamic request batching for optimal throughput
- **Memory Optimization**: Efficient KV cache management with swapping

### 7. Distributed Computing (`vllm/distributed/`)

Handles communication between distributed workers:

```mermaid
graph TB
    subgraph "Communication"
        COMM[communication_op.py<br/>Communication Operations]
        DEV[device_communicators/<br/>Device Communication]
        PAR[parallel_state.py<br/>Parallel State Management]
    end
    
    subgraph "KV Transfer"
        KVT[kv_transfer/<br/>KV Cache Transfer]
        KVE[kv_events.py<br/>Event Management]
        PIPE[kv_pipe/<br/>Pipeline Transfer]
    end
    
    subgraph "Parallelism Types"
        TP[Tensor Parallelism]
        PP[Pipeline Parallelism]
        DP[Data Parallelism]
    end
    
    COMM --> TP
    COMM --> PP
    COMM --> DP
    
    PAR --> TP
    PAR --> PP
    
    KVT --> PP
    KVE --> KVT
    PIPE --> KVT
```

**Key Functions:**
- **Tensor Parallelism**: Distribute model weights across multiple GPUs
- **Pipeline Parallelism**: Distribute model layers across multiple GPUs
- **KV Cache Transfer**: Efficient transfer of attention caches between stages
- **Communication Optimization**: Minimize inter-GPU communication overhead

### 8. Specialized Features

#### Speculative Decoding (`vllm/spec_decode/`)

```mermaid
graph LR
    subgraph "Speculative Decoding"
        BASE[proposer_worker_base.py<br/>Base Proposer]
        NGRAM[ngram_worker.py<br/>N-gram Proposer]
        MLP[mlp_speculator_worker.py<br/>MLP Speculator]
        MEDUSA[medusa_worker.py<br/>Medusa Head]
        MULTI[multi_step_worker.py<br/>Multi-step]
        SMALL[smaller_tp_proposer_worker.py<br/>Smaller TP]
    end
    
    BASE --> NGRAM
    BASE --> MLP
    BASE --> MEDUSA
    BASE --> MULTI
    BASE --> SMALL
```

#### LoRA Support (`vllm/lora/`)

```mermaid
graph LR
    subgraph "LoRA Components"
        REQ[request.py<br/>LoRA Requests]
        WM[worker_manager.py<br/>Worker Management]
        LAYERS[layers.py<br/>LoRA Layers]
        MOD[models.py<br/>LoRA Models]
    end
    
    REQ --> WM
    WM --> LAYERS
    LAYERS --> MOD
```

#### Multimodal Support (`vllm/multimodal/`)

```mermaid
graph LR
    subgraph "Multimodal"
        PROC[processing.py<br/>Input Processing]
        REG[registry.py<br/>Modality Registry]
        UTILS[utils.py<br/>Utilities]
        BASE[base.py<br/>Base Classes]
    end
    
    BASE --> REG
    REG --> PROC
    PROC --> UTILS
```

## Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant API/LLM
    participant Engine
    participant Scheduler
    participant Executor
    participant Worker
    participant Model
    
    Client->>API/LLM: Request
    API/LLM->>Engine: Process Request
    Engine->>Scheduler: Schedule Request
    Scheduler->>Scheduler: Batch & Allocate Memory
    Scheduler->>Executor: Execute Batch
    Executor->>Worker: Distribute Work
    Worker->>Model: Run Inference
    Model->>Worker: Model Output
    Worker->>Executor: Collect Results
    Executor->>Scheduler: Return Results
    Scheduler->>Engine: Process Output
    Engine->>API/LLM: Format Response
    API/LLM->>Client: Response
```

## Key Innovations

1. **PagedAttention**: Virtual memory-style management for attention computation
2. **Continuous Batching**: Dynamic request batching for optimal throughput
3. **Prefix Caching**: Intelligent caching of common prefixes
4. **Multi-backend Support**: Flexible attention backend selection
5. **Distributed Execution**: Efficient tensor and pipeline parallelism
6. **Speculative Decoding**: Accelerated generation with speculation
7. **Multi-modal Support**: Unified interface for text, image, and other modalities

## Performance Features

- **High Throughput**: State-of-the-art serving performance
- **Memory Efficiency**: Optimal GPU memory utilization
- **Low Latency**: Fast model execution with CUDA graphs
- **Scalability**: Support for large-scale distributed deployment
- **Flexibility**: Multiple deployment options and serving interfaces

This architecture enables VLLM to achieve superior performance while maintaining ease of use and flexibility for various deployment scenarios.