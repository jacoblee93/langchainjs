import { Tool, VectorStoreQATool } from "../../tools/index.js";
import { VectorStore } from "../../../vectorstores/index.js";
import { Toolkit } from "../base.js";
import { BaseLanguageModel } from "../../../base_language/index.js";
import { CreatePromptArgs, ZeroShotAgent } from "../../mrkl/index.js";
import { VECTOR_PREFIX, VECTOR_ROUTER_PREFIX } from "./prompt.js";
import { getSuffixForLLMType } from "../../mrkl/prompt.js";
import { LLMChain } from "../../../chains/index.js";
import { AgentExecutor } from "../../executor.js";

export interface VectorStoreInfo {
  vectorStore: VectorStore;
  name: string;
  description: string;
}

export class VectorStoreToolkit extends Toolkit {
  tools: Tool[];

  llm: BaseLanguageModel;

  constructor(vectorStoreInfo: VectorStoreInfo, llm: BaseLanguageModel) {
    super();
    const description = VectorStoreQATool.getDescription(
      vectorStoreInfo.name,
      vectorStoreInfo.description
    );
    this.llm = llm;
    this.tools = [
      new VectorStoreQATool(vectorStoreInfo.name, description, {
        vectorStore: vectorStoreInfo.vectorStore,
        llm: this.llm,
      }),
    ];
  }
}

export class VectorStoreRouterToolkit extends Toolkit {
  tools: Tool[];

  vectorStoreInfos: VectorStoreInfo[];

  llm: BaseLanguageModel;

  constructor(vectorStoreInfos: VectorStoreInfo[], llm: BaseLanguageModel) {
    super();
    this.llm = llm;
    this.vectorStoreInfos = vectorStoreInfos;
    this.tools = vectorStoreInfos.map((vectorStoreInfo) => {
      const description = VectorStoreQATool.getDescription(
        vectorStoreInfo.name,
        vectorStoreInfo.description
      );
      return new VectorStoreQATool(vectorStoreInfo.name, description, {
        vectorStore: vectorStoreInfo.vectorStore,
        llm: this.llm,
      });
    });
  }
}

export function createVectorStoreAgent(
  llm: BaseLanguageModel,
  toolkit: VectorStoreToolkit,
  args?: CreatePromptArgs
) {
  const {
    prefix = VECTOR_PREFIX,
    suffix = getSuffixForLLMType(llm._llmType()),
    inputVariables = ["input", "agent_scratchpad"],
  } = args ?? {};
  const { tools } = toolkit;
  const prompt = ZeroShotAgent.createPrompt(tools, {
    prefix,
    suffix,
    inputVariables,
  });
  const chain = new LLMChain({ prompt, llm });
  const agent = new ZeroShotAgent({
    llmChain: chain,
    allowedTools: tools.map((t) => t.name),
  });
  return AgentExecutor.fromAgentAndTools({
    agent,
    tools,
    returnIntermediateSteps: true,
  });
}

export function createVectorStoreRouterAgent(
  llm: BaseLanguageModel,
  toolkit: VectorStoreRouterToolkit,
  args?: CreatePromptArgs
) {
  const {
    prefix = VECTOR_ROUTER_PREFIX,
    suffix = getSuffixForLLMType(llm._llmType()),
    inputVariables = ["input", "agent_scratchpad"],
  } = args ?? {};
  const { tools } = toolkit;
  const prompt = ZeroShotAgent.createPrompt(tools, {
    prefix,
    suffix,
    inputVariables,
  });
  const chain = new LLMChain({ prompt, llm });
  const agent = new ZeroShotAgent({
    llmChain: chain,
    allowedTools: tools.map((t) => t.name),
  });
  return AgentExecutor.fromAgentAndTools({
    agent,
    tools,
    returnIntermediateSteps: true,
  });
}
