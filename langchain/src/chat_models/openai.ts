import { OpenAI, ClientOptions } from "openai";
import { getEnvironmentVariable } from "../util/env.js";
import {
  AzureOpenAIInput,
  OpenAICallOptions,
  OpenAIChatInput,
} from "../types/openai-types.js";
// import fetchAdapter from "../util/axios-fetch-adapter.js";
// import type { StreamingAxiosConfiguration } from "../util/axios-types.js";
import { BaseChatModel, BaseChatModelParams } from "./base.js";
import {
  AIMessage,
  BaseMessage,
  ChatGeneration,
  ChatMessage,
  ChatResult,
  HumanMessage,
  MessageType,
  SystemMessage,
} from "../schema/index.js";
import { getModelNameForTiktoken } from "../base_language/count_tokens.js";
import { CallbackManagerForLLMRun } from "../callbacks/manager.js";
import { promptLayerTrackRequest } from "../util/prompt-layer.js";
import { StructuredTool } from "../tools/base.js";
import { formatToOpenAIFunction } from "../tools/convert_to_openai.js";
// import { getEndpoint, OpenAIEndpointConfig } from "../util/azure.js";

export { OpenAICallOptions, OpenAIChatInput, AzureOpenAIInput };

interface TokenUsage {
  completionTokens?: number;
  promptTokens?: number;
  totalTokens?: number;
}

interface OpenAILLMOutput {
  tokenUsage: TokenUsage;
}

function messageTypeToOpenAIRole(
  type: MessageType
): 'system' | 'user' | 'assistant' | 'function' {
  switch (type) {
    case "system":
      return "system";
    case "ai":
      return "assistant";
    case "human":
      return "user";
    case "function":
      return "function";
    default:
      throw new Error(`Unknown message type: ${type}`);
  }
}

function openAIResponseToChatMessage(
  message: OpenAI.Chat.ChatCompletion.Choice.Message
): BaseMessage {
  switch (message.role) {
    case "user":
      return new HumanMessage(message.content || "");
    case "assistant":
      return new AIMessage(message.content || "", {
        function_call: message.function_call,
      });
    case "system":
      return new SystemMessage(message.content || "");
    default:
      return new ChatMessage(message.content || "", message.role ?? "unknown");
  }
}

export interface ChatOpenAICallOptions extends OpenAICallOptions {
  function_call?: OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestNonStreaming.FunctionCallOption;
  functions?: OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestNonStreaming.Function[];
  tools?: StructuredTool[];
  promptIndex?: number;
}

/**
 * Wrapper around OpenAI large language models that use the Chat endpoint.
 *
 * To use you should have the `openai` package installed, with the
 * `OPENAI_API_KEY` environment variable set.
 *
 * To use with Azure you should have the `openai` package installed, with the
 * `AZURE_OPENAI_API_KEY`,
 * `AZURE_OPENAI_API_INSTANCE_NAME`,
 * `AZURE_OPENAI_API_DEPLOYMENT_NAME`
 * and `AZURE_OPENAI_API_VERSION` environment variable set.
 * `AZURE_OPENAI_BASE_PATH` is optional and will override `AZURE_OPENAI_API_INSTANCE_NAME` if you need to use a custom endpoint.
 *
 * @remarks
 * Any parameters that are valid to be passed to {@link
 * https://platform.openai.com/docs/api-reference/chat/create |
 * `openai.createChatCompletion`} can be passed through {@link modelKwargs}, even
 * if not explicitly available on this class.
 */
export class ChatOpenAI
  extends BaseChatModel
  implements OpenAIChatInput, AzureOpenAIInput
{
  declare CallOptions: ChatOpenAICallOptions;

  get callKeys(): (keyof ChatOpenAICallOptions)[] {
    return [
      ...(super.callKeys as (keyof ChatOpenAICallOptions)[]),
      "options",
      "function_call",
      "functions",
      "tools",
      "promptIndex",
    ];
  }

  lc_serializable = true;

  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      openAIApiKey: "OPENAI_API_KEY",
      azureOpenAIApiKey: "AZURE_OPENAI_API_KEY",
    };
  }

  get lc_aliases(): Record<string, string> {
    return {
      modelName: "model",
      openAIApiKey: "openai_api_key",
      azureOpenAIApiVersion: "azure_openai_api_version",
      azureOpenAIApiKey: "azure_openai_api_key",
      azureOpenAIApiInstanceName: "azure_openai_api_instance_name",
      azureOpenAIApiDeploymentName: "azure_openai_api_deployment_name",
    };
  }

  temperature = 1;

  topP = 1;

  frequencyPenalty = 0;

  presencePenalty = 0;

  n = 1;

  logitBias?: Record<string, number>;

  modelName = "gpt-3.5-turbo";

  modelKwargs?: OpenAIChatInput["modelKwargs"];

  stop?: string[];

  timeout?: number;

  streaming = false;

  maxTokens?: number;

  openAIApiKey?: string;

  azureOpenAIApiVersion?: string;

  azureOpenAIApiKey?: string;

  azureOpenAIApiInstanceName?: string;

  azureOpenAIApiDeploymentName?: string;

  azureOpenAIBasePath?: string;

  private client: OpenAI;

  private clientConfig: ClientOptions;

  constructor(
    fields?: Partial<OpenAIChatInput> &
      Partial<AzureOpenAIInput> &
      BaseChatModelParams & {
        configuration?: ClientOptions;
      },
    /** @deprecated */
    configuration?: ClientOptions
  ) {
    super(fields ?? {});

    this.openAIApiKey =
      fields?.openAIApiKey ?? getEnvironmentVariable("OPENAI_API_KEY");

    this.azureOpenAIApiKey =
      fields?.azureOpenAIApiKey ??
      getEnvironmentVariable("AZURE_OPENAI_API_KEY");

    if (!this.azureOpenAIApiKey && !this.openAIApiKey) {
      throw new Error("OpenAI or Azure OpenAI API key not found");
    }

    this.azureOpenAIApiInstanceName =
      fields?.azureOpenAIApiInstanceName ??
      getEnvironmentVariable("AZURE_OPENAI_API_INSTANCE_NAME");

    this.azureOpenAIApiDeploymentName =
      fields?.azureOpenAIApiDeploymentName ??
      getEnvironmentVariable("AZURE_OPENAI_API_DEPLOYMENT_NAME");

    this.azureOpenAIApiVersion =
      fields?.azureOpenAIApiVersion ??
      getEnvironmentVariable("AZURE_OPENAI_API_VERSION");

    this.azureOpenAIBasePath =
      fields?.azureOpenAIBasePath ??
      getEnvironmentVariable("AZURE_OPENAI_BASE_PATH");

    this.modelName = fields?.modelName ?? this.modelName;
    this.modelKwargs = fields?.modelKwargs ?? {};
    this.timeout = fields?.timeout;

    this.temperature = fields?.temperature ?? this.temperature;
    this.topP = fields?.topP ?? this.topP;
    this.frequencyPenalty = fields?.frequencyPenalty ?? this.frequencyPenalty;
    this.presencePenalty = fields?.presencePenalty ?? this.presencePenalty;
    this.maxTokens = fields?.maxTokens;
    this.n = fields?.n ?? this.n;
    this.logitBias = fields?.logitBias;
    this.stop = fields?.stop;

    this.streaming = fields?.streaming ?? false;

    if (this.azureOpenAIApiKey) {
      if (!this.azureOpenAIApiInstanceName && !this.azureOpenAIBasePath) {
        throw new Error("Azure OpenAI API instance name or base path not found");
      }
      if (!this.azureOpenAIApiDeploymentName) {
        throw new Error("Azure OpenAI API deployment name not found");
      }
      if (!this.azureOpenAIApiVersion) {
        throw new Error("Azure OpenAI API version not found");
      }
    }

    this.clientConfig = {
      apiKey: this.openAIApiKey,
      ...configuration,
      ...fields?.configuration,
    };
  }

  /**
   * Get the parameters used to invoke the model
   */
  invocationParams(
    options?: this["ParsedCallOptions"]
  ): Omit<OpenAI.Chat.CompletionCreateParams, "messages"> {
    return {
      model: this.modelName,
      temperature: this.temperature,
      top_p: this.topP,
      frequency_penalty: this.frequencyPenalty,
      presence_penalty: this.presencePenalty,
      max_tokens: this.maxTokens === -1 ? undefined : this.maxTokens,
      n: this.n,
      logit_bias: this.logitBias,
      stop: options?.stop ?? this.stop,
      stream: this.streaming,
      functions:
        options?.functions ??
        (options?.tools
          ? options?.tools.map(formatToOpenAIFunction)
          : undefined),
      function_call: options?.function_call,
      ...this.modelKwargs,
    };
  }

  /** @ignore */
  _identifyingParams(): ClientOptions & Omit<OpenAI.Chat.CompletionCreateParams, "messages"> & { model_name: string } {
    return {
      model_name: this.modelName,
      ...this.invocationParams(),
      ...this.clientConfig,
    };
  }

  /**
   * Get the identifying parameters for the model
   */
  identifyingParams() {
    return this._identifyingParams();
  }

  /** @ignore */
  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const tokenUsage: TokenUsage = {};
    const params = this.invocationParams(options);
    const messagesMapped: OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestNonStreaming.Message[] = messages.map(
      (message) => ({
        role: messageTypeToOpenAIRole(message._getType()),
        content: message.content,
        name: message.name,
        function_call: message.additional_kwargs
          .function_call as OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestNonStreaming.Message.FunctionCall,
      })
    );

    const data = await this.completionWithRetry({
      ...params,
      messages: messagesMapped,
    }, { signal: options.signal ?? options.options?.signal, promptIndex: options.promptIndex }, runManager);

    const {
      completion_tokens: completionTokens,
      prompt_tokens: promptTokens,
      total_tokens: totalTokens,
    } = data.usage ?? {};

    if (completionTokens) {
      tokenUsage.completionTokens =
        (tokenUsage.completionTokens ?? 0) + completionTokens;
    }

    if (promptTokens) {
      tokenUsage.promptTokens = (tokenUsage.promptTokens ?? 0) + promptTokens;
    }

    if (totalTokens) {
      tokenUsage.totalTokens = (tokenUsage.totalTokens ?? 0) + totalTokens;
    }

    const generations: ChatGeneration[] = [];
    for (const part of data.choices) {
      const text = part.message?.content ?? "";
      generations.push({
        text,
        message: openAIResponseToChatMessage(
          part.message ?? { role: "assistant" }
        ),
      });
    }
    return {
      generations,
      llmOutput: { tokenUsage },
    };
  }

  async getNumTokensFromMessages(messages: BaseMessage[]): Promise<{
    totalCount: number;
    countPerMessage: number[];
  }> {
    let totalCount = 0;
    let tokensPerMessage = 0;
    let tokensPerName = 0;

    // From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    if (getModelNameForTiktoken(this.modelName) === "gpt-3.5-turbo") {
      tokensPerMessage = 4;
      tokensPerName = -1;
    } else if (getModelNameForTiktoken(this.modelName).startsWith("gpt-4")) {
      tokensPerMessage = 3;
      tokensPerName = 1;
    }

    const countPerMessage = await Promise.all(
      messages.map(async (message) => {
        const textCount = await this.getNumTokens(message.content);
        const roleCount = await this.getNumTokens(
          messageTypeToOpenAIRole(message._getType())
        );
        const nameCount =
          message.name !== undefined
            ? tokensPerName + (await this.getNumTokens(message.name))
            : 0;
        const count = textCount + tokensPerMessage + roleCount + nameCount;

        totalCount += count;
        return count;
      })
    );

    totalCount += 3; // every reply is primed with <|start|>assistant<|message|>

    return { totalCount, countPerMessage };
  }

  /** @ignore */
  async completionWithRetry(
    params: OpenAI.Chat.CompletionCreateParams,
    options: { signal?: AbortSignal, promptIndex?: number },
    runManager?: CallbackManagerForLLMRun
  ): Promise<any> {
    if (!this.client) {
      const openAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        basePath: this.clientConfig.basePath,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const clientConfig = {
        ...this.clientConfig,
        // maxRetries: 1,
        // baseURL: endpoint,
        // baseOptions: {
          // timeout: this.timeout,
          // ...this.clientConfig.baseOptions,
        // },
      };

      this.client = new OpenAI(clientConfig);
    }

    console.log({ signal: options.signal })
    const completionRequest = params.stream ?
      async () => {
        const stream = await this.client.chat.completions.create({
          ...params,
          stream: true,
        }, { signal: options.signal, maxRetries: 1 });
        let finalResponse;
        for await (const part of stream) {
          if (!finalResponse) {
            finalResponse = {
              id: part.id,
              object: part.object,
              created: part.created,
              model: part.model,
              choices: [] as Record<string, any>[],
            };
          }
          for (const choice of part.choices) {
            let finalResponseChoice = finalResponse.choices.find((c) => c.index === choice.index);
            if (!finalResponseChoice) {
              finalResponseChoice = {
                index: choice.index,
                finish_reason: choice.finish_reason ?? undefined,
              };
              finalResponse.choices.push(finalResponseChoice);
            }
            if (!finalResponseChoice.message) {
              finalResponseChoice.message = {
                role: choice.delta.role,
                content: ""
              };
            }
            if (choice.delta.function_call && !finalResponseChoice.message.function_call) {
              finalResponseChoice.message.function_call = {
                name: "",
                arguments: "",
              };
            }
            finalResponseChoice.message.content += choice.delta.content ?? "";
            if (finalResponseChoice.message.function_call) {
              finalResponseChoice.message.function_call.name += choice.delta.function_call?.name ?? "";
              finalResponseChoice.message.function_call.arguments += choice.delta.function_call?.arguments ?? "";
            }
            // eslint-disable-next-line no-void
            void runManager?.handleLLMNewToken(
              choice.delta?.content ?? "",
              {
                prompt: options.promptIndex ?? 0,
                completion: choice.index,
              }
            );
          }
        }
        return finalResponse as OpenAI.Chat.ChatCompletion;
      } : async () => {
        return this.client.chat.completions.create({
          ...params,
          stream: false,
        }, { signal: options.signal, maxRetries: 1 });
      }

  //   const axiosOptions = {
  //     adapter: isNode() ? undefined : fetchAdapter,
  //     // ...this.clientConfig.baseOptions,
  //     ...options,
  //   } as StreamingAxiosConfiguration;
  //   if (this.azureOpenAIApiKey) {
  //     axiosOptions.headers = {
  //       "api-key": this.azureOpenAIApiKey,
  //       ...axiosOptions.headers,
  //     };
  //     axiosOptions.params = {
  //       "api-version": this.azureOpenAIApiVersion,
  //       ...axiosOptions.params,
  //     };
  //   }
    return this.caller.call(completionRequest);
  }

  _llmType() {
    return "openai";
  }

  /** @ignore */
  _combineLLMOutput(...llmOutputs: OpenAILLMOutput[]): OpenAILLMOutput {
    return llmOutputs.reduce<{
      [key in keyof OpenAILLMOutput]: Required<OpenAILLMOutput[key]>;
    }>(
      (acc, llmOutput) => {
        if (llmOutput && llmOutput.tokenUsage) {
          acc.tokenUsage.completionTokens +=
            llmOutput.tokenUsage.completionTokens ?? 0;
          acc.tokenUsage.promptTokens += llmOutput.tokenUsage.promptTokens ?? 0;
          acc.tokenUsage.totalTokens += llmOutput.tokenUsage.totalTokens ?? 0;
        }
        return acc;
      },
      {
        tokenUsage: {
          completionTokens: 0,
          promptTokens: 0,
          totalTokens: 0,
        },
      }
    );
  }
}

export class PromptLayerChatOpenAI extends ChatOpenAI {
  promptLayerApiKey?: string;

  plTags?: string[];

  returnPromptLayerId?: boolean;

  constructor(
    fields?: ConstructorParameters<typeof ChatOpenAI>[0] & {
      promptLayerApiKey?: string;
      plTags?: string[];
      returnPromptLayerId?: boolean;
    }
  ) {
    super(fields);

    this.promptLayerApiKey =
      fields?.promptLayerApiKey ??
      (typeof process !== "undefined"
        ? // eslint-disable-next-line no-process-env
          process.env?.PROMPTLAYER_API_KEY
        : undefined);
    this.plTags = fields?.plTags ?? [];
    this.returnPromptLayerId = fields?.returnPromptLayerId ?? false;
  }

  async _generate(
    messages: BaseMessage[],
    options?: string[] | this["CallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const requestStartTime = Date.now();

    let parsedOptions: this["CallOptions"];
    if (Array.isArray(options)) {
      parsedOptions = { stop: options } as this["CallOptions"];
    } else if (options?.timeout && !options.signal) {
      parsedOptions = {
        ...options,
        signal: AbortSignal.timeout(options.timeout),
      };
    } else {
      parsedOptions = options ?? {};
    }

    const generatedResponses = await super._generate(
      messages,
      parsedOptions,
      runManager
    );
    const requestEndTime = Date.now();

    const _convertMessageToDict = (message: BaseMessage) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let messageDict: Record<string, any>;

      if (message._getType() === "human") {
        messageDict = { role: "user", content: message.content };
      } else if (message._getType() === "ai") {
        messageDict = { role: "assistant", content: message.content };
      } else if (message._getType() === "function") {
        messageDict = { role: "assistant", content: message.content };
      } else if (message._getType() === "system") {
        messageDict = { role: "system", content: message.content };
      } else if (message._getType() === "generic") {
        messageDict = {
          role: (message as ChatMessage).role,
          content: message.content,
        };
      } else {
        throw new Error(`Got unknown type ${message}`);
      }

      return messageDict;
    };

    const _createMessageDicts = (
      messages: BaseMessage[],
      callOptions?: this["CallOptions"]
    ) => {
      const params = {
        ...this.invocationParams(),
        model: this.modelName,
      };

      if (callOptions?.stop) {
        if (Object.keys(params).includes("stop")) {
          throw new Error("`stop` found in both the input and default params.");
        }
      }
      const messageDicts = messages.map((message) =>
        _convertMessageToDict(message)
      );
      return messageDicts;
    };

    for (let i = 0; i < generatedResponses.generations.length; i += 1) {
      const generation = generatedResponses.generations[i];
      const messageDicts = _createMessageDicts(messages, parsedOptions);

      let promptLayerRequestId: string | undefined;
      const parsedResp = [
        {
          content: generation.text,
          role: messageTypeToOpenAIRole(generation.message._getType()),
        },
      ];

      const promptLayerRespBody = await promptLayerTrackRequest(
        this.caller,
        "langchain.PromptLayerChatOpenAI",
        messageDicts,
        this._identifyingParams(),
        this.plTags,
        parsedResp,
        requestStartTime,
        requestEndTime,
        this.promptLayerApiKey
      );

      if (this.returnPromptLayerId === true) {
        if (promptLayerRespBody.success === true) {
          promptLayerRequestId = promptLayerRespBody.request_id;
        }

        if (
          !generation.generationInfo ||
          typeof generation.generationInfo !== "object"
        ) {
          generation.generationInfo = {};
        }

        generation.generationInfo.promptLayerRequestId = promptLayerRequestId;
      }
    }

    return generatedResponses;
  }
}
