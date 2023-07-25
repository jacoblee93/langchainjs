import { AxiosRequestConfig } from "axios";
import { OpenAI, ClientOptions } from "openai";

import type { BaseLanguageModelCallOptions } from "../base_language/index.js";
import type { RequestOptions } from "openai/core.js";
import type { Agent } from "openai/_shims/agent.js";

export declare interface OpenAIBaseInput {
  /** Sampling temperature to use */
  temperature: number;

  /**
   * Maximum number of tokens to generate in the completion. -1 returns as many
   * tokens as possible given the prompt and the model's maximum context size.
   */
  maxTokens?: number;

  /** Total probability mass of tokens to consider at each step */
  topP: number;

  /** Penalizes repeated tokens according to frequency */
  frequencyPenalty: number;

  /** Penalizes repeated tokens */
  presencePenalty: number;

  /** Number of completions to generate for each prompt */
  n: number;

  /** Dictionary used to adjust the probability of specific tokens being generated */
  logitBias?: Record<string, number>;

  /** Whether to stream the results or not. Enabling disables tokenUsage reporting */
  streaming: boolean;

  /** Model name to use */
  modelName: string;

  /** Holds any additional parameters that are valid to pass to {@link
   * https://platform.openai.com/docs/api-reference/completions/create |
   * `openai.createCompletion`} that are not explicitly specified on this class.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelKwargs?: Record<string, any>;

  /** List of stop words to use when generating */
  stop?: string[];

  /**
   * Timeout to use when making requests to OpenAI.
   */
  timeout?: number;

  /**
   * API key to use when making requests to OpenAI. Defaults to the value of
   * `OPENAI_API_KEY` environment variable.
   */
  openAIApiKey?: string;
}

export interface OpenAICallOptions extends BaseLanguageModelCallOptions {
  /**
   * Additional options to pass to the underlying axios request.
   */
  options?: AxiosRequestConfig;
}

/**
 * Input to OpenAI class.
 */
export declare interface OpenAIInput extends OpenAIBaseInput {
  /** Generates `bestOf` completions server side and returns the "best" */
  bestOf?: number;

  /** Batch size to use when passing multiple documents to generate */
  batchSize: number;
}

export type OpenAIChatMessage = OpenAI.Chat.ChatCompletion.Choice.Message;

export type OpenAIChatMessageParam = OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestNonStreaming.Message;

export type OpenAIFunction = OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestStreaming.Function;

export type OpenAIFunctionCall = OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestStreaming.Message.FunctionCall;

export type OpenAIFunctionCallOption = OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestStreaming.FunctionCallOption;

// Include some supported Axios legacy fields
export type OpenAIClientOptions = Omit<ClientOptions, "maxRetries"> & {
  organization?: string;
  /** @deprecated Use "baseURL" instead */
  basePath?: string;
  /** @deprecated Set corresponding top-level fields instead */
  baseOptions?: {
    /** @deprecated Use top-level "defaultHeaders" instead */
    headers?: Record<string, string | null | undefined>;
    /** @deprecated Use top-level "defaultQuery" instead */
    params?: Record<string, string | undefined>;
    /** @deprecated Use top-level "httpAgent" instead */
    httpAgent: Agent;
    /** @deprecated Use top-level "httpAgent" instead */
    httpsAgent: Agent;
  }
};

export type OpenAIRequestOptions = Omit<RequestOptions, "maxRetries"> & {
  /** @deprecated Use "query" instead */
  params?: Record<string, string | undefined>;
}

export interface OpenAIChatInput extends OpenAIBaseInput {
  /** ChatGPT messages to pass as a prefix to the prompt */
  prefixMessages?: OpenAI.Chat.ChatCompletion[];
}

export declare interface AzureOpenAIInput {
  /**
   * API version to use when making requests to Azure OpenAI.
   */
  azureOpenAIApiVersion?: string;

  /**
   * API key to use when making requests to Azure OpenAI.
   */
  azureOpenAIApiKey?: string;

  /**
   * Azure OpenAI API instance name to use when making requests to Azure OpenAI.
   * this is the name of the instance you created in the Azure portal.
   * e.g. "my-openai-instance"
   * this will be used in the endpoint URL: https://my-openai-instance.openai.azure.com/openai/deployments/{DeploymentName}/
   */
  azureOpenAIApiInstanceName?: string;

  /**
   * Azure OpenAI API deployment name to use for completions when making requests to Azure OpenAI.
   * This is the name of the deployment you created in the Azure portal.
   * e.g. "my-openai-deployment"
   * this will be used in the endpoint URL: https://{InstanceName}.openai.azure.com/openai/deployments/my-openai-deployment/
   */
  azureOpenAIApiDeploymentName?: string;

  /**
   * Azure OpenAI API deployment name to use for embedding when making requests to Azure OpenAI.
   * This is the name of the deployment you created in the Azure portal.
   * This will fallback to azureOpenAIApiDeploymentName if not provided.
   * e.g. "my-openai-deployment"
   * this will be used in the endpoint URL: https://{InstanceName}.openai.azure.com/openai/deployments/my-openai-deployment/
   */
  azureOpenAIApiEmbeddingsDeploymentName?: string;

  /**
   * Azure OpenAI API deployment name to use for completions when making requests to Azure OpenAI.
   * Completions are only available for gpt-3.5-turbo and text-davinci-003 deployments.
   * This is the name of the deployment you created in the Azure portal.
   * This will fallback to azureOpenAIApiDeploymentName if not provided.
   * e.g. "my-openai-deployment"
   * this will be used in the endpoint URL: https://{InstanceName}.openai.azure.com/openai/deployments/my-openai-deployment/
   */
  azureOpenAIApiCompletionsDeploymentName?: string;

  /**
   * Custom endpoint for Azure OpenAI API. This is useful in case you have a deployment in another region.
   * e.g. setting this value to "https://westeurope.api.cognitive.microsoft.com/openai/deployments"
   * will be result in the endpoint URL: https://westeurope.api.cognitive.microsoft.com/openai/deployments/{DeploymentName}/
   */
  azureOpenAIBasePath?: string;
}
