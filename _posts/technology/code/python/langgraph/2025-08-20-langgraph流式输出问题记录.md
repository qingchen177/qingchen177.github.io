---
layout: post
title: LangGraph流式输出问题记录
tags: [LangGraph]
categories: ["Python"]
---

## 前情

我克隆项目代码安装环境后，跑项目中间碰到的问题如下：

对图设置流式输出的模式`stream_mode`有下面这几种，项目里面配置了`messages`和`updates`，原理上在配置了`messages`后会把图里面有关大模型输出的内容全部吐出来，但是我这边实际上并没有输出`messages`，但是`updates`是正常的，一开始以为是代码哪里写的有问题，但是回过神来我是刚拷的公司项目代码什么都没改，同事本地也能跑，就定位到是环境的问题了（langgraph包的问题）。

> https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.StreamMode
>
> ```python
> StreamMode = Literal[
>     "values",
>     "updates",
>     "checkpoints",
>     "tasks",
>     "debug",
>     "messages",
>     "custom",
> ]
> ```

```python
    async for stream_mode, chunk in graph.astream(
            {"some_text": "original text"},
            stream_mode=["values", "updates", "messages"],
            config=config
    ):
        print(f"Stream mode: {stream_mode}，chunk：{chunk}")
        print("\n")
```

## 目前的解决办法

降低`langgraph`的版本

测试下来`0.5.x`和`0.6.x`都不行

最后安装`0.4.5`解决

```shell
pip install langgraph==0.4.5
pip install langgraph-prebuilt==0.5.0
```

我看到相关issue已经有人提出来了:[在这里](https://github.com/langchain-ai/langgraph/issues/5951)

## 插曲

发现不同平台提供的不同模型的流式返回也不一样：

比如gemini和硅基流动这两个，我让他们给我一个4字词语二者流式返回如下：

```python
# gemini
Stream mode: messages，chunk：(AIMessageChunk(content='运转如常', additional_kwargs={}, response_metadata={'safety_ratings': [], 'usage_metadata': {}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash'}, id='run--8d9060e0-7f53-4813-953d-77bf79525b97', usage_metadata={'input_tokens': 22, 'output_tokens': 3, 'total_tokens': 1101, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 1076}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:c516e5ad-20d4-58da-24ce-ce6524c362cf|task_node:c8f26ac3-6e77-e501-9dd7-5e9cdada09f2', 'checkpoint_ns': 'human_node:c516e5ad-20d4-58da-24ce-ce6524c362cf|task_node:c8f26ac3-6e77-e501-9dd7-5e9cdada09f2', 'ls_provider': 'google_vertexai', 'ls_model_name': 'gemini-2.5-flash', 'ls_model_type': 'chat', 'ls_temperature': None})

# siliconflow
Stream mode: messages，chunk：(AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 284, 'total_tokens': 312, 'input_token_details': {}, 'output_token_details': {'reasoning': 284}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})

...

Stream mode: messages，chunk：(AIMessageChunk(content='\n\n', additional_kwargs={}, response_metadata={}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 288, 'total_tokens': 316, 'input_token_details': {}, 'output_token_details': {'reasoning': 287}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})


Stream mode: messages，chunk：(AIMessageChunk(content='按', additional_kwargs={}, response_metadata={}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 289, 'total_tokens': 317, 'input_token_details': {}, 'output_token_details': {'reasoning': 287}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})


Stream mode: messages，chunk：(AIMessageChunk(content='部', additional_kwargs={}, response_metadata={}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 290, 'total_tokens': 318, 'input_token_details': {}, 'output_token_details': {'reasoning': 287}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})


Stream mode: messages，chunk：(AIMessageChunk(content='就', additional_kwargs={}, response_metadata={}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 291, 'total_tokens': 319, 'input_token_details': {}, 'output_token_details': {'reasoning': 287}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})


Stream mode: messages，chunk：(AIMessageChunk(content='班', additional_kwargs={}, response_metadata={}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 292, 'total_tokens': 320, 'input_token_details': {}, 'output_token_details': {'reasoning': 287}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})


Stream mode: messages，chunk：(AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'Qwen/Qwen3-32B'}, id='run--9a4bb426-9da3-4307-8001-708b6b653556', usage_metadata={'input_tokens': 28, 'output_tokens': 292, 'total_tokens': 320, 'input_token_details': {}, 'output_token_details': {'reasoning': 287}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'checkpoint_ns': 'human_node:1c4f6ac7-4460-ed5d-f773-8ec231b97b59|task_node:2a668166-512c-249b-bb01-9226ae6e9aff', 'ls_provider': 'openai', 'ls_model_name': 'Qwen/Qwen3-32B', 'ls_model_type': 'chat', 'ls_temperature': 0.9})

```

siliconflow一直返回空，真是害人呐！！！
