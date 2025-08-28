---
layout: post
title: LangGraph persistence
tags: [langgraph]
categories: ["人工智能"]
---

## LangGraph persistence

关于langgraph的持久化学习记录

### checkpoint

首先`compile a graph with a checkpointer`（简写成ckpter吧），这个ckpter会在每个步骤去保存一个ckpt，ckpts都是存放到一个线程`thread`里面。

`StateSnapshot`对象保存每个步骤的图状态

### thread

调用graph就需要指定线程id

```python
{"configurable": {"thread_id": "1"}}
```

### 示例代码

主要通过代码看下可以实现的功能吧

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add] # 这里加上了reducer函数

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 先指定线程id
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)

# get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)

# 拿全部的state记录
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))

# play-back 回放
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.invoke(None, config=config)

# 更新某个节点的参数
# update_state(config,values，as_node)
# config如果只传thread_id则更新当前的state，加上ckpt会fork选择的检查点
# value就是state的值，这里的就是有一点要注意会根据参数的reducer去覆盖或者追加
graph.update_state(config, {"foo": 2, "bar": ["c"]})
# 这里我更新完state就变成了{'foo': 2, 'bar': ['a', 'b', 'c']}
# as_node的用法，指定node名称在node节点更新数据
graph.update_state(config, {"foo": 2, "bar": ["d"]}, as_node="node_a")
graph.get_state(config)
```

### memory

因为ckpter不能跨线程，所以来个memory存储

```python
from langgraph.store.memory import InMemoryStore
import uuid

store = InMemoryStore()
user_id = "qingchen"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"fans": "jay chou"}
store.put(namespace_for_memory, memory_id, memory)
memories = store.search(namespace_for_memory)
memories[-1]
```

#### 语义搜索

```python
# 语义搜索
from langchain_qingchen.data.embedding.QingchenEmbeddings import QingchenEmbeddings

store = InMemoryStore(
    index={
        "embed": QingchenEmbeddings(),  # Embedding provider
        "dims": 1024,  # Embedding dimensions
        "fields": ["fans", "$"]  # Fields to embed
    }
)
user_id = "qingchen"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"fans": "I'm jay chou's fans"}
store.put(namespace_for_memory, memory_id, memory)
# Find memories about jay chou
# (This can be done after putting memories into the store)
memories = store.search(
    namespace_for_memory,
    query="Whose fan is it?",
    limit=3  # Return top 3 matches
)
memories
```

