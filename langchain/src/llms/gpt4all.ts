import { GPT4All as GPT4AllClient } from "gpt4all";

import { LLM, BaseLLMParams } from "./base.js";

export interface GPT4AllInput {
  // These are the only two models supported by the gpt4all-ts library currently.
  // TODO: The gpt4all TS package does not currently export an input type - if they do, use it here.
  model: "gpt4all-lora-unfiltered-quantized" | "gpt4all-lora-quantized";
  forceDownload?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  decoderConfig?: Record<string, any>;
}

export class GPT4All extends LLM {
  private model: GPT4AllInput["model"];

  private forceDownload: boolean;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private decoderConfig: Record<string, any>;

  private gpt4AllClient: GPT4AllClient;

  private isInitialized = false;

  constructor(fields: GPT4AllInput & BaseLLMParams) {
    super(fields);

    this.model = fields.model;
    this.forceDownload = fields.forceDownload ?? false;
    this.decoderConfig = fields.decoderConfig ?? {};
    this.gpt4AllClient = new GPT4AllClient(
      this.model,
      this.forceDownload,
      this.decoderConfig
    );
  }

  _llmType() {
    return "gpt4all";
  }

  async initialize() {
    await this.gpt4AllClient.init();
    this.isInitialized = true;
  }

  async _call(prompt: string, _stop?: string[]): Promise<string> {
    if (!this.isInitialized) {
      await this.gpt4AllClient.init();
      this.isInitialized = true;
    }

    await this.gpt4AllClient.open();

    const output = await this.caller.call(
      this.gpt4AllClient.prompt.bind(null, prompt)
    );

    this.gpt4AllClient.close();
    return output;
  }
}
