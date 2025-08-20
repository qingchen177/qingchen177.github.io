---
layout: post
title: langgraph interrupt记录
tags: [langgraph]
categories: ["人工智能"]
---

## [interrupt](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Interrupt)

https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/#pause-using-interrupt

简单的来说，`graph`中运行到`interrupt`就会暂停，直到遇到`command`后继续运行。

> 关键在于这一段：
>
> Interrupts resemble Python's input() function in terms of developer experience, but they do not automatically resume execution from the interruption point. Instead, they rerun the entire node where the interrupt was used. For this reason, interrupts are typically best placed at the start of a node or in a dedicated node.
> 就开发者体验而言，中断类似于 Python 的 input() 函数，但它们不会自动从中断点恢复执行。相反，它们会重新运行发生中断的整个节点。因此，中断通常最好放置在节点的起始位置或专用节点中。

这里说会重新运行中断的那个节点，我这里提出一个问题，如果中断前节点中有大模型的结果返回，恢复后还会不会再去调一次大模型重新生成呢？看下下面的测试代码：

```python
import os
from typing import TypedDict
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.func import task
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

from langchain_qingchen.chat.QingchenModel import QingchenModel

count = 0


class State(TypedDict):
    some_text: str


async def human_node(state: State):
    global count
    count += 1
    print(f"inter human node!!! count：{count}")
    item = await task_node()
    print("item:", item)
    value = interrupt("提示你要怎么做")
    print("value:", value)
    return {
        "some_text": value
    }

model = QingchenModel()


@task
async def task_node() -> str:
    content = ''
    async for chunk in model.astream("来个4字成语，表达你现在的状态，只回答成语就好，不需要其他额外解释"):
        if hasattr(chunk, 'content'):
            chunk_content = chunk.content
            if isinstance(chunk_content, str):
                content += chunk_content
            else:
                content += str(chunk_content)
        else:
            content += str(chunk)
    return content


graph_builder = StateGraph(State)
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}



# streaming
async def main():
    async for stream_mode, chunk in graph.astream(
            {"some_text": "original text"},
            stream_mode=["values", "updates", "messages"],
            config=config
    ):
        print(f"Stream mode: {stream_mode}，chunk：{chunk}")

    print("+++++++++++++ resume +++++++++++++")
    async for stream_mode, chunk in graph.astream(
            Command(resume="Edited text"),
            stream_mode=["values", "messages", "updates"],
            config=config
    ):
        print(f"Stream mode: {stream_mode}，chunk：{chunk}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

```

输出如下：

```python
Stream mode: values，chunk：{'some_text': 'original text'}

inter human node!!! count：1
Stream mode: messages，chunk：(AIMessageChunk(content='待命而动', additional_kwargs={}, response_metadata={'safety_ratings': [], 'usage_metadata': {}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash'}, id='run--ee8964f4-9e29-44d3-99c6-56ee8d83d440', usage_metadata={'input_tokens': 22, 'output_tokens': 4, 'total_tokens': 832, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 806}}), {'langgraph_step': 1, 'langgraph_node': 'task_node', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', ('__pregel_pull', 'human_node'), 0, True), 'langgraph_checkpoint_ns': 'human_node:db512a53-c8b0-97de-8c14-3472df064d8b|task_node:0e361e7b-88a9-e9a7-4ec4-add42bcba6a9', 'checkpoint_ns': 'human_node:db512a53-c8b0-97de-8c14-3472df064d8b|task_node:0e361e7b-88a9-e9a7-4ec4-add42bcba6a9', 'ls_provider': 'google_vertexai', 'ls_model_name': 'gemini-2.5-flash', 'ls_model_type': 'chat', 'ls_temperature': None})

item: 待命而动
Stream mode: updates，chunk：{'task_node': '待命而动'}

Stream mode: updates，chunk：{'__interrupt__': (Interrupt(value='提示你要怎么做？', resumable=True, ns=['human_node:db512a53-c8b0-97de-8c14-3472df064d8b']),)}

+++++++++++++ resume +++++++++++++
Stream mode: values，chunk：{'some_text': 'original text'}

inter human node!!! count：2
item: 待命而动
value: Edited text
Stream mode: updates，chunk：{'task_node': '待命而动'}


Stream mode: updates，chunk：{'human_node': {'some_text': 'Edited text'}}


Stream mode: values，chunk：{'some_text': 'Edited text'}
```

发现只输出了一次`Stream mode: messages，chunk：(AIMessageChunk(content='待命而动'`，但是确实count变成2了。大模型的第二次并没有调用而是直接用的`graph`里面的`state`缓存。

