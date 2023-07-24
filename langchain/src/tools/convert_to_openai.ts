import { zodToJsonSchema } from "zod-to-json-schema";
import { OpenAI } from "openai";

import { StructuredTool } from "./base.js";

export function formatToOpenAIFunction(
  tool: StructuredTool
): OpenAI.Chat.CompletionCreateParams.CreateChatCompletionRequestStreaming.Function {
  return {
    name: tool.name,
    description: tool.description,
    parameters: zodToJsonSchema(tool.schema),
  };
}
