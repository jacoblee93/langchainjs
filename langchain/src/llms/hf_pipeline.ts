import { LLM, BaseLLMParams } from "./base.js";

export interface HFPipelineInput {
  /** Model to use */
  model: string;

  /** Sampling temperature to use */
  temperature?: number;

  /**
   * Maximum number of tokens to generate in the completion.
   */
  maxTokens?: number;

  /** Total probability mass of tokens to consider at each step */
  topP?: number;

  /** Integer to define the top tokens considered within the sample operation to create new text. */
  topK?: number;

  /** Penalizes repeated tokens according to frequency */
  frequencyPenalty?: number;

  /** API key to use. */
  apiKey?: string;
}

export class HuggingFacePipeline extends LLM implements HFPipelineInput {

  _llmType() {
    return "huggingface_pipeline";
  }

  /** @ignore */
  async _call(prompt: string, _stop?: string[]): Promise<string> {
    const { pipeline, env } = await HuggingFacePipeline.imports();
    // const res = await this.caller.call(hf.textGeneration.bind(hf), {
    //   model: this.model,
    //   parameters: {
    //     // make it behave similar to openai, returning only the generated text
    //     return_full_text: false,
    //     temperature: this.temperature,
    //     max_new_tokens: this.maxTokens,
    //     top_p: this.topP,
    //     top_k: this.topK,
    //     repetition_penalty: this.frequencyPenalty,
    //   },
    //   inputs: prompt,
    // });
    // return res.generated_text;
  }

  /** @ignore */
  static async imports(): Promise<typeof import("@xenova/transformers")> {
    try {
      const transformers = await import("@xenova/transformers");
      return transformers;
    } catch (e) {
      throw new Error(
        "Please install Transformers.js as a dependency with, e.g. `yarn add @xenova/transformers`"
      );
    }
  }

}