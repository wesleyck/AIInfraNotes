# 1.  整体架构

git commit hash
```
# 2026.01.06
46255effe4166e7d433766dd98237ebfaadbc82e
```

# 2. 学习目标

## 2.1 调度Scheduler
### 2.1.1 chunk prefill 逻辑

**设计原理**
包装成跟正常的请求唯一的区别就是增加了一个无法decode false的boolean，标识这条请求还没完成prefill，其实就长请求切块成chunk，然后每次利用cache reuse，其他的id，kv的信息都不变，这样的话能直接复用，调度上考虑放在队列的最前部分，这样避免新请求过来占资源产生死锁
1. 调度上id那些info，chunk完成前请求一致
2. kv cache上跑完一轮没有到next token，会标识到一个dummy的请求，在token pool or page table写到一个特殊的位置


### 2.1.2 prefill & decode 调度设计

prefill优先的话，请求一直有，在遇到资源瓶颈前是不是 decode会一直饥饿？
- 目前代码prefill和decode分开调度，也就是一个batch里面全部都是prefill或者全部都是decode，遇到请求一直来的情况下，再达到资源瓶颈（kv cache容量）的稳定期前，确实prefill会不断打断decode的生成，导致一直饥饿，所以有个to do后续支持decode first或者vllm这种的类似混合调度（一个batch里面有prefill 和decode）
- 目前的打断方式其实跟chunk 与否没啥区别，因为chunk prefill的请求会一直高优打断，导致其实chunk prefll的原始收益（降低峰值显存占用、降低tpot）里面只有降低峰值显存有效，也是就不是decode见缝插针

### 2.1.3 overlap调度

overlap的是cpu的执行和gpu的执行，异构硬件，不用相互等待，传统pipline的是cpu调度等gpu处理、然后gpu处理等cpu调度和预处理，相互等待，导致gpu idle

一般调度流程：b1 cpu调度&数据预处理 -> b1 model forward -> b1 cpu数据预处理

这里的核心设计原理在于：双流设计，通过一个engine stream做gpu的forward，另一个engine做预处理（数据拷贝）和后处理，线程中的cpu同步处理很快、涉及到gpu的操作支持提交任务是异步的，所以能这样做，然后engine的输入出入依赖通过流同步来完成，schedule需要的后处理数据通过event带出来解决数据依赖，这样起步两个batch的情况下就会有two batch overlap cpu处理与gpu处理，从而完全流水化

在gpu处理非常快的情况下比如gb200-300的时候，设计前提得gpu耗时大于cpu处理可能不一定成立，目前卡型没那么高暂时不用太考虑

## 2.2  kv cache 

### 2.2.1 基本原理

kv cache基本原理、为什么chunk与否不影响精度？

radix cacahe相比正常的结构有啥区别？


### 2.2.2 申请与evict

如何申请、如何驱逐

## 2.3 attention
### 2.3.1 整体设计架构

只是针对flash attention 和flash infer包装了一层对吗？

整体如何涉及的，参数如何更新的


涉及到cuda graph如何做的
### 2.3.2 flash infer vs flash attention

区别是什么？


metadata是什么？需要的具体的推理的信息是什么？


rope什么时候代入进去

## 2.4 engine

### 2.4.1 cuda graph

解决什么问题？padding的多计算和性能的tradeoff如何衡量，什么情况下失效


如何进行cuda graph的组织的，如何capture 和replay，哪些部分进行了cuda graph


模型推理的过程是不是除了attention就是gemm了？


## 2.5 并行：TP

如何通信的，不同卡如何初始化的，怎么区分的


tp如何切分、切分的是什么？模型的哪些需要切分？attention还是mlp？还是其他别的地方？


为什么不会影响精度？

## 2.6 python tricks

现代python项目组织形式：依赖、安装方式、


test如何组织的


为什么每个文件夹里面都有一个__init__，里面的__all__= 是什么意思，还有的包里面有个__main__然后里面就有一个函数，这个是干嘛的


这里的装饰器@nvtx_annotate会影响性能吗，默认开启吗？






