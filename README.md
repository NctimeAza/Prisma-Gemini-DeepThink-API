# Prisma Gemini DeepThink API

**本项目处于早期开发阶段，很多功能都不完善。**

基于[yeahhe365/Prisma](https://github.com/yeahhe365/Prisma)前端项目做的纯后端二次开发。Prisma前端不在本仓库范围内，这里只有Python/FastAPI实现的多专家深度推理流水线。对外暴露OpenAI兼容接口，下游任何能支持OAI API的客户端都能直接接。

主要针对**Gemini 3.1 Pro**调优测试，其余模型（Flash、Kimi K2.5、Deepseek等）兼容性比较蛆。整体设计**重文轻理**，创意写作、翻译润色、深度分析等主观型任务是常用场景。

---

## 原理

将单一问题分解为多专家并行处理，经审查循环迭代后综合输出。流水线分4阶段：

```
REQ -> [Manager规划] -> [Expert并行执行] -> [Manager审查] <-> [迭代] -> [Synthesis综合] -> RSP
```

- Manager规划阶段：分析你的问题，输出结构化JSON，包含N个专家配置（角色、描述、温度、具体任务prompt）。
- Expert并行执行阶段：各专家独立并发调用LLM（有防429），互不共享上下文，看不到其它专家输出。
- Manager审查阶段：评估专家输出，对每个专家做出keep（一般是满意了）/iterate（需要改进）/delete（回复质量太低且毫无改进空间，踢出上下文）决策。未通过则分配新一轮专家继续迭代，直至满意或达到`max_rounds`上限。
- Synthesis综合阶段：汇总全部轮次专家结果，流式输出最终回答。

所有LLM调用经统一Provider抽象层路由，支持Gemini原生协议与OpenAI兼容协议两种上游。每个阶段有独立的thinking budget（由虚拟模型配置中的`planning_level`/`expert_level`/`synthesis_level`控制，似乎对OAI模型不兼容），温度亦可按阶段锁定，避免有些提供商的模型只支持特定温度。

## 一次DeepThink流程

1. 下游POST `/v1/chat/completions`，携带虚拟模型名（例如`gemini-3.1-pro-deepthink-high`）。
2. 路由层通过`resolve_model`将虚拟模型解析为：实际模型、Manager/Synthesis专用模型、各阶段thinking level、max_rounds、provider、各阶段温度覆盖。生成`DeepThinkConfig`交给编排器。
3. Manager规划阶段：Manager调用LLM，返回`AnalysisResult`（结构化JSON），内含专家配置列表。若返回空列表则注入兜底专家。
4. Expert并行执行阶段：按配置构建`ExpertResult`列表，`asyncio.gather`并发执行。每个专家独立拿到对话上下文+自己的任务prompt+图片（如有），非流式调用LLM。执行结果（content/thoughts/status）回写到ExpertResult。
5. Review Loop：若`enable_recursive_loop=true`且`round < max_rounds`：Manager拿到全部专家输出进行审查，对每个专家输出做action决策：
   - `keep`: 保留原样
   - `iterate`: 标记原专家`context_status=iterated`，基于原输出+审查意见构建迭代prompt，分配新专家继续改进
   - `delete`: 标记`context_status=deleted`，原内容替换为删除说明，踢出后续上下文
   - 同时可分配全新方向的`refined_experts`

   新一轮专家再次并发执行，循环直至`satisfied=true`或达到轮次上限。
6. **Synthesis** -- 收集全部轮次专家结果+审查记录，流式调用LLM输出最终回答。`reasoning_content`和`content`通过SSE并行推送给客户端。
7. 全程维护**Checkpoint**（`DeepThinkCheckpoint`），各阶段状态持久化到磁盘。中途断连后可通过`!deepthink_continue <resume_id>`从断点恢复。

## 配置相关

```powershell
py -m pip install -r requirements.txt
copy .env.example .env
# 根据个人情况配置好.env，启动
py main.py
```

关键配置参见`.env.example`，注释应该够详尽了，大多数配置默认都可，此处仅提几个容易踩坑的：

- `OPENAI_BASE_URL`：须带`/v1`后缀。留空走官方。
- `MAX_ROUNDS`：默认审查轮数。虚拟模型配置里的`max_rounds`字段会覆盖它。比如high档默认5轮，extra档10轮
- `LLM_REQUEST_DELAY_MIN/MAX`：请求间随机延迟，防429或者风控。不好说有没有用
- `SSE_HEARTBEAT_INTERVAL`：SSE心跳间隔。用NewAPI等中转时如果频繁断连，缩短此值
- `LLM_REQUEST_TIMEOUT`：LLM请求超时时间，单位秒。一般默认600即可，这是谷歌API的非流最大超时时间了

### 虚拟模型

内置`gemini-3.1-pro-deepthink-{minimal,low,medium,high,extra}`五档预设。通过`VIRTUAL_MODELS_FILE`指向JSON文件可新增或覆盖默认模型。每个虚拟模型可独立指定：实际模型、Manager/Synthesis专用模型、provider，以及更细粒度的 `manager_provider` / `expert_provider` / `synthesis_provider`，各阶段thinking level与温度覆盖、max_rounds。格式详见`.env.example`。

### 多Provider

通过环境变量注册自定义provider：`PROVIDER_<NAME>_API_KEY` + `PROVIDER_<NAME>_BASE_URL`，然后在虚拟模型配置中引用`"provider": "<name>"`即可。

### 注意

- OAI兼容接口不支持搜索：GoogleSearch工具（硬编码开启的）仅Gemini原生协议可用。非Gemini provider下无联网检索能力。
- 非Gemini模型的兼容性问题：项目依赖模型具备较强的JSON结构化输出能力（Manager规划/审查阶段需要严格JSON）。实测DS官API不支持`response_format`，直接400不可用；Kimi K2.5官API不用温度1会报400，且有时JSON遵循性差。
- Thinking Budget对非Gemini模型无效：`thinkingConfig`参数会被静默忽略，实际效果取决于模型自身。

## TODO

- [ ] 架构重构，改变基本Deepthink流程来降本增效
- [ ] 智能路由，可让规划或审查阶段分配成本更低的小模型来解决简单问题

---

*详细的流程图、代码映射表和Checkpoint状态机参见项目内`Doc.md`。*

## License

本项目采用 [MIT License](./LICENSE)。

本项目说明中提到基于 `yeahhe365/Prisma` 做二次开发；其上游仓库同样为 MIT 许可证。若仓库中包含或后续引入了上游代码，请保留对应版权声明与 MIT 许可文本。
