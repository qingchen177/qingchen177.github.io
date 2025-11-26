---
layout: post
title: Langgraph Command 的妙用
tags: [LangGraph]
categories: ["Python"]
---

## 场景

做了一个支付的 Agent ：

主要功能是用户选产品、下单、付款、交付的一系列功能。



环节：

用户与 Agent 交互后创建订单环节，

通过 interrupt 返回订单信息，等待用户支付，支付消息(data)通过回调返回，agent 在此暂停等待回调数据；

用户输入了信息，继续正常对话；

收到回调则去交付相关产品信息



问题在于：

用户继续聊天后支付了，回调需要走到支付信息对应的 interrupt 的节点，去执行对应的流程



## 解决方法

我一开始想到是的 checkpoint 恢复，后来发现恢复就是再跑一次，传参进去压根没啥影响



command 我一直用，我以为必须要先触发 interrupt 才能去 resume

后面我直接尝试都传，嘿，您瞧怎么着，真行嘿



关键代码：

```python
Command(
            goto="interrupt_node",  # 直接去对应的节点
            resume={"data": "我付完款了，哥们，真不赖!"}  # 带上 resume 这样就不会 interrupt 了， 会按照顺序执行了
        )
```



直接看代码吧！

## 代码

```python
from pprint import pprint

from langchain_core.language_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command


class State(MessagesState):
    interrupt: int
    chat: int


def mock_llm(state: State):
    return {"messages": [{"role": "ai", "content": "下单成功，请支付..."}]}


def interrupt_node(state: State):
    payload = interrupt("订单创建完成，等待哥们你的支付...")
    data = payload.get("data")  # 支付回调
    message = payload.get("message")  # 用户继续沟通

    if data:
        return Command(
            goto="data_node",
            update={
                "interrupt": state.get("interrupt", 0) + 1,
                "messages": [{"role": "ai", "content": data}]
            }
        )

    if message:
        return {
            "chat": state.get("chat", 0) + 1,
            "messages": [
                {"role": "human", "content": message},
                {"role": "ai", "content": "包有的，哥们！"}
            ]
        }

    # Error
    return {"messages": [{"role": "human", "content": "不是哥们，传参呢？胡闹！"}, ]}


def data_node(state: State):
    fake_message = FakeListChatModel(responses=["哥们已经在 🫵 给你发货了"]).invoke("")
    return {"messages": [fake_message]}


graph = StateGraph(State)

graph.add_node(mock_llm)
graph.add_node(interrupt_node)
graph.add_node(data_node)

graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", "interrupt_node")
graph.add_edge("interrupt_node", END)
graph.add_edge("data_node", END)

checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

for model, value in graph.stream(
        {"messages": [{"role": "user", "content": "哥们，我来买一张方大同的黑胶唱片"}]},
        stream_mode=["updates", "checkpoints"],
        config=config
):
    print("*" * 100)
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)
    print()

print("=" * 100)
print("state: ", graph.get_state(config=config))
print()

# 恢复中断
for model, value in graph.stream(
        Command(resume={"message": "你真有啊？"}),
        stream_mode=["updates", "checkpoints"],
        config=config
):
    print("*" * 100)
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)
    print()

print("=" * 100)
print("state: ", graph.get_state(config=config))
print()

for model, value in graph.stream(
        Command(
            goto="interrupt_node",
            resume={"data": "我付完款了，哥们，真不赖!"}
        ),
        stream_mode=["updates", "checkpoints"],
        config=config
):
    print("*" * 100)
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)
    print()

```

输出：

```text
****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2b-68ee-bfff-e57d121d9c93'}},
 'parent_config': None,
 'values': {'messages': []},
 'metadata': {'source': 'input',
              'step': -1,
              'parents': {}},
 'next': ['__start__'],
 'tasks': [{'id': '03ce38ff-7777-e58f-3803-d697e595fe8d',
            'name': '__start__',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c2b-68ee-bfff-e57d121d9c93'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5')]},
 'metadata': {'source': 'loop',
              'step': 0,
              'parents': {}},
 'next': ['mock_llm'],
 'tasks': [{'id': '7e2a12db-50d7-7328-4344-7cef5ad24630',
            'name': 'mock_llm',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'mock_llm': {'messages': [{'role': 'ai',
                            'content': '下单成功，请支付...'}]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b')]},
 'metadata': {'source': 'loop',
              'step': 1,
              'parents': {}},
 'next': ['interrupt_node'],
 'tasks': [{'id': '2ff3e45a-a474-fb08-a15d-631aac7ea187',
            'name': 'interrupt_node',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'__interrupt__': (Interrupt(value='订单创建完成，等待哥们你的支付...',
                             id='b63ac8c675718737076485b99fa18474'),)}

====================================================================================================
state:  StateSnapshot(values={'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'), AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b')]}, next=('interrupt_node',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-11-25T11:25:58.337438+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}}, tasks=(PregelTask(id='2ff3e45a-a474-fb08-a15d-631aac7ea187', name='interrupt_node', path=('__pregel_pull', 'interrupt_node'), error=None, interrupts=(Interrupt(value='订单创建完成，等待哥们你的支付...', id='b63ac8c675718737076485b99fa18474'),), state=None, result=None),), interrupts=(Interrupt(value='订单创建完成，等待哥们你的支付...', id='b63ac8c675718737076485b99fa18474'),))

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'parent_config': {'configurable': {'thread_id': '1',
                                    'checkpoint_ns': '',
                                    'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b')]},
 'metadata': {'source': 'loop',
              'step': 1,
              'parents': {}},
 'next': ['interrupt_node'],
 'tasks': [{'id': '2ff3e45a-a474-fb08-a15d-631aac7ea187',
            'name': 'interrupt_node',
            'interrupts': ({'value': '订单创建完成，等待哥们你的支付...',
                            'id': 'b63ac8c675718737076485b99fa18474'},),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'interrupt_node': {'chat': 1,
                    'messages': [{'role': 'human',
                                  'content': '你真有啊？'},
                                 {'role': 'ai',
                                  'content': '包有的，哥们！'}]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75')],
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 2,
              'parents': {}},
 'next': [],
 'tasks': []}

====================================================================================================
state:  StateSnapshot(values={'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'), AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'), HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'), AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75')], 'chat': 1}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2025-11-25T11:25:58.338770+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}}, tasks=(), interrupts=())

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}},
 'parent_config': {'configurable': {'thread_id': '1',
                                    'checkpoint_ns': '',
                                    'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75')],
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 2,
              'parents': {}},
 'next': ['interrupt_node'],
 'tasks': [{'id': '672d0452-4f2d-cf6b-5499-3fb9c79845b8',
            'name': 'interrupt_node',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'interrupt_node': {'interrupt': 1,
                    'messages': [{'role': 'ai',
                                  'content': '我付完款了，哥们，真不赖!'}]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c34-64da-8003-a7facb4cab51'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75'),
                         AIMessage(content='我付完款了，哥们，真不赖!', additional_kwargs={}, response_metadata={}, id='d3797fd1-c200-45dc-b19a-b92ef187f06e')],
            'interrupt': 1,
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 3,
              'parents': {}},
 'next': ['data_node'],
 'tasks': [{'id': '07eb0f07-5c1f-58f5-e985-33092ee626a9',
            'name': 'data_node',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'data_node': {'messages': [AIMessage(content='哥们已经在 🫵 给你发货了', additional_kwargs={}, response_metadata={}, id='run--50c0d268-00c9-4c7f-b076-663988e10803-0')]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c36-6212-8004-acae72ab008d'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c34-64da-8003-a7facb4cab51'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75'),
                         AIMessage(content='我付完款了，哥们，真不赖!', additional_kwargs={}, response_metadata={}, id='d3797fd1-c200-45dc-b19a-b92ef187f06e'),
                         AIMessage(content='哥们已经在 🫵 给你发货了', additional_kwargs={}, response_metadata={}, id='run--50c0d268-00c9-4c7f-b076-663988e10803-0')],
            'interrupt': 1,
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 4,
              'parents': {}},
 'next': [],
 'tasks': []}
```

